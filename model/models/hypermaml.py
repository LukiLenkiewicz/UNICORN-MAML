from collections import OrderedDict

import torch
import torch.nn as nn


def inner_train_step(model, support_data, hn, args):
    """ Inner training step procedure. """
    # obtain final prediction
    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)
    onehot = torch.zeros(len(support_data), args.way)
    onehot[torch.arange(len(support_data)), label] = 1


    params = OrderedDict(model.named_parameters())

    embeddings, logits = model(support_data, params, embedding_and_logits=True)

    enhanced_embeddings = torch.cat([embeddings, logits, onehot.cuda()], dim=1)

    avg_enhanced_embeddings = []

    for l in set(label.cpu().numpy()):
        avg_enhanced_embeddings.append(
            enhanced_embeddings[label==l].mean(dim=0)
        )

    avg_enhanced_embeddings = torch.stack(avg_enhanced_embeddings)


    hn_out = hn(avg_enhanced_embeddings)

    weigths_update = hn_out[:, :-1]
    bias_update = hn_out[:, -1]

    assert params["fc.weight"].shape == weigths_update.shape
    assert params["fc.bias"].shape == bias_update.shape


    params["fc.weight"] = params["fc.weight"] + weigths_update
    params["fc.bias"] = params["fc.bias"] + bias_update

    return params

class HN(nn.Module):
    def __init__(self, args, hdim: int):
        super().__init__()
        self.head_len = 3
        self.hidden_size = 256

        layers = [
            nn.Linear(hdim + 2 * args.way, self.hidden_size),
            nn.ReLU()
        ]
        for i in range(self.head_len - 1):
            layers.extend([nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()])
        layers.append(nn.Linear(self.hidden_size, hdim + 1))

        self.hn = nn.Sequential(*layers)

    def forward(self, support_embeddings: torch.Tensor):
        return self.hn(support_embeddings)


class HyperMAML(nn.Module):

    def __init__(self, args):
        super().__init__()

        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12_maml import ResNetMAML
            self.encoder = ResNetMAML(dropblock_size=args.dropblock_size)
        elif args.backbone_class == "Conv4":
            hdim = 64
            from model.networks.convnet_maml import ConvNet
            self.encoder = ConvNet()
        else:
            raise ValueError('')

        self.args = args
        self.hdim = hdim
        self.encoder.fc = nn.Linear(hdim, args.way)
        self.hn = HN(args=args, hdim=hdim)

    def forward(self, data_shot, data_query):
        # update with gradient descent
        updated_params = inner_train_step(model=self.encoder, support_data=data_shot, hn=self.hn,  args=self.args)

        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis

    def forward_eval(self, data_shot, data_query):
        # update with gradient descent
        self.train()
        updated_params = inner_train_step(model=self.encoder, support_data=data_shot, hn=self.hn,  args=self.args)

        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis_query
