from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP


class BinaryHyperMAML(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.train_lr = 0.01
        self.encoder = self.init_encoder(args)
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()
        self.encoder.fc = nn.Linear(self.hdim, args.way)
        self.hn = self.init_hypernetwork()
        from model.networks.convnet import ConvNet as SupportConvNet
        self.support_encoder = SupportConvNet(avg_pool=(args.dataset != "cross_char"))

    def init_encoder(self, args):
        if args.backbone_class == 'Res12':
            self.hdim = 640
            from model.networks.res12_maml import ResNetMAML
            self.encoder = ResNetMAML(dropblock_size=args.dropblock_size)
        elif args.backbone_class == "Conv4":
            self.hdim = 64
            from model.networks.convnet_maml import ConvNet
            self.encoder = ConvNet(avg_pool=(args.dataset != "cross_char"))
        else:
            raise ValueError('')
        return self.encoder

    def init_hypernetwork(self):
        self.embedding_size = (self.hdim + 2 * self.args.way)*self.args.way
        shapes = list(map(lambda layer: list(layer.shape), self.encoder.parameters()))

        # shapes = [list(layer.shape) for layer in self.encoder.parameters()]
        hypernet_layers = [self.args.hm_hn_len for _ in range(self.args.hm_hn_width)]
        return ChunkedHMLP(shapes, uncond_in_size=self.embedding_size, cond_in_size=0, chunk_emb_size=self.args.bm_chunk_emb_size,
            layers=hypernet_layers, chunk_size=self.args.bm_chunk_size, num_cond_embs=1)

    def forward(self, data_shot, data_query):
        updated_params = self.inner_train_step(support_data=data_shot,  args=self.args)

        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis

    def forward_eval(self, data_shot, data_query):
        self.train()
        updated_params = self.inner_train_step(support_data=data_shot, args=self.args)

        self.eval()
        with torch.no_grad():
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis_query

    def inner_train_step(self, support_data, args):
        """ Inner training step procedure. """
        # obtain final prediction
        label = torch.arange(args.way).repeat(args.shot)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        onehot = torch.zeros(len(support_data), args.way)
        onehot[torch.arange(len(support_data)), label] = 1


        params = OrderedDict(self.encoder.named_parameters())
        # embeddings, logits = self.encoder(support_data, params, embedding_and_logits=True)
        
        # embeddings = self.support_encoder(support_data)
        # logits = self.encoder.fc(embeddings)
        embeddings, logits = self.encoder(support_data, embedding_and_logits=True)

        enhanced_embeddings = torch.cat([embeddings, logits, onehot.cuda()], dim=1)

        avg_enhanced_embeddings = []

        for l in set(label.cpu().numpy()):
            avg_enhanced_embeddings.append(
                enhanced_embeddings[label==l].mean(dim=0)
            )

        avg_enhanced_embeddings = torch.stack(avg_enhanced_embeddings)

        avg_enhanced_embeddings = avg_enhanced_embeddings.reshape(1, -1)
        hn_out = self.hn(avg_enhanced_embeddings)

        for name, updates in zip(params, hn_out):
            params[name] = params[name] * updates

        updated_params = self.apply_maml_updates(support_data, label)

        return updated_params

    def apply_maml_updates(self, support_data, support_data_labels):
        updated_params = OrderedDict(self.encoder.named_parameters())
        for task_step in range(self.args.inner_iters):
            scores = self.encoder(support_data)
            loss = self.loss_fn(scores, support_data_labels)
            self.maml_step(loss, updated_params)

        return updated_params

    def maml_step(self, loss, params):
        name_list, tensor_list = zip(*params.items())
        grads = torch.autograd.grad(loss, tensor_list, create_graph=False)
        updated_params = OrderedDict()
        for name, param, grad in zip(name_list, tensor_list, grads):
            updated_params[name] = param - self.train_lr * grad

        return updated_params