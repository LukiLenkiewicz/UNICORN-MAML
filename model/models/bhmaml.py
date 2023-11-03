from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from hypnettorch.hnets.chunked_mlp_hnet import ChunkedHMLP

from model.networks.convnet_maml import ConvNet
from model.networks.res12 import ResNet
from model.networks.res12_maml import ResNetMAML
from model.networks.convnet import ConvNet as SupportConvNet
from model.models.bhmaml_utils import Binarizer, percentile


class BinaryHyperMAML(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.loss_fn = nn.CrossEntropyLoss()

        self.encoder, self.support_encoder = self.init_encoders(args)
        self.encoder.fc = nn.Linear(self.hdim, args.way)
        self.hn = self.init_hypernetwork()

    def init_encoders(self, args):
        encoder = self.init_query_encoder(args)
        support_encoder = self.init_support_encoder(args)
        
        return encoder, support_encoder 

    def init_query_encoder(self, args):
        if args.backbone_class == 'Res12':
            self.hdim = 640
            encoder = ResNetMAML(dropblock_size=args.dropblock_size)
        elif args.backbone_class == "Conv4":
            self.hdim = 64
            encoder = ConvNet(avg_pool=(args.dataset != "cross_char"))
        else:
            raise ValueError('')
        
        return encoder
    
    def init_support_encoder(self, args):
        if args.backbone_class == 'Res12':
            support_encoder = ResNet(dropblock_size=args.dropblock_size)
        elif args.backbone_class == "Conv4":
            support_encoder = SupportConvNet(avg_pool=(args.dataset != "cross_char"))
        else:
            raise ValueError('')

        return support_encoder

    def init_hypernetwork(self):
        self.embedding_size = (self.hdim + 2 * self.args.way)*self.args.way
        shapes = list(map(lambda layer: list(layer.shape), self.encoder.parameters()))

        # shapes = [list(layer.shape) for layer in self.encoder.parameters()]
        hypernet_layers = [self.args.hm_hn_len for _ in range(self.args.hm_hn_width)]
        return ChunkedHMLP(shapes, uncond_in_size=self.embedding_size, cond_in_size=0, chunk_emb_size=self.args.bm_chunk_emb_size,  # TODO: check uncond in size and cond in size
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
        label = self.prepare_label(args)

        params = OrderedDict(self.encoder.named_parameters())

        if self.args.bm_maml_first:
            params = self.apply_maml_updates(params, support_data, label)
            updated_params = self.apply_hypermaml_updates(params, support_data, label)
        else:
            params = self.apply_hypermaml_updates(params, support_data, label)
            updated_params = self.apply_maml_updates(params, support_data, label)

        return updated_params
    
    def prepare_label(self, args):
        label = torch.arange(args.way).repeat(args.shot)
        if torch.cuda.is_available():
            return label.type(torch.cuda.LongTensor)
        else:
            return label.type(torch.LongTensor)

    def apply_hypermaml_updates(self, params, support_data, label):
        onehot = torch.zeros(len(support_data), self.args.way)
        onehot[torch.arange(len(support_data)), label] = 1

        embeddings = self.support_encoder(support_data)
        logits = self.encoder.fc(embeddings)
        logits = logits.detach()

        enhanced_embeddings = torch.cat([embeddings, logits, onehot.cuda()], dim=1)
        avg_enhanced_embeddings = []

        for l in set(label.cpu().numpy()):
            avg_enhanced_embeddings.append(
                enhanced_embeddings[label==l].mean(dim=0)
            )

        avg_enhanced_embeddings = torch.stack(avg_enhanced_embeddings)

        avg_enhanced_embeddings = avg_enhanced_embeddings.reshape(1, -1)
        hn_out = self.hn(avg_enhanced_embeddings)

        hn_out = self.apply_binarizer(hn_out)

        for name, updates in zip(params, hn_out):
            params[name] = params[name] * updates

        return params

    def apply_binarizer(self, delta_params):
        for i in range(len(delta_params)):
            delta_params[i] = torch.sigmoid(delta_params[i])

        params_flat = [param.clone().detach().reshape(-1) for param in delta_params]
        concat = torch.cat(params_flat)
        num_params = len(concat)

        k_val = torch.quantile(concat, self.args.bm_mask_size).item()

        ones_ = 0
        for i in range(len(delta_params)):
            delta_params[i] = Binarizer.apply(delta_params[i], k_val)
            ones_ += torch.sum(delta_params[i]).item()
        
        return delta_params

    def apply_maml_updates(self, params, support_data, support_data_labels):
        for task_step in range(self.args.inner_iters):
            scores = self.encoder(support_data, params)
            # loss = F.cross_entropy(scores, support_data_labels)
            loss = self.loss_fn(scores, support_data_labels) 
            params = self.maml_step(loss, params)

        return params

    def maml_step(self, loss, params):
        name_list, tensor_list = zip(*params.items())
        grads = torch.autograd.grad(loss, tensor_list, create_graph=True, allow_unused=True)
        updated_params = OrderedDict()
        for name, param, grad in zip(name_list, tensor_list, grads):
            updated_params[name] = param - self.args.gd_lr * grad

        return updated_params
