from email.mime import base
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import itertools
from collections import OrderedDict
from model.utils import count_acc
from operator import itemgetter
from model.models.ranker import Ranker
from lapsolver import solve_dense as lap_solve_dense


def update_params(loss, params, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad

    return updated_params


def get_enhanced_embeddings(model, support_data, args):
    label = torch.arange(args.way).repeat(args.shot)

    onehot = torch.zeros(len(support_data), args.way)
    onehot[torch.arange(len(support_data)), label] = 1

    params = OrderedDict(model.named_parameters())
    embeddings, logits = model(support_data, params, embedding_and_logits=True)
    enhanced_embeddings = torch.cat([embeddings, logits, onehot.cuda()], dim=1)

    avg_enhanced_embeddings = []

    for l in label.tolist():
        avg_enhanced_embeddings.append(
            enhanced_embeddings[label == l].mean(dim=0)
        )

    avg_enhanced_embeddings = torch.stack(avg_enhanced_embeddings)
    return avg_enhanced_embeddings


def inner_train_step(model, support_data, args, permutation):
    """ Inner training step procedure. """
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    label = torch.tensor(permutation).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)

    for _ in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.cross_entropy(ypred, label)
        updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)
    return updated_params


class RankerAttention(nn.Module):
    def __init__(self, args, hdim, last_activation_fn='sigmoid', num_heads: int = 8):
        super().__init__()

        self.args = args
        self.hdim = hdim
        self.depth = args.ranker_depth
        self.width = args.ranker_width

        supports_layers = [nn.Linear(
            hdim + (args.way * 2),  # enchanced embeddings
            self.width)]
        params_layers = [nn.Linear(hdim, self.width)]

        for _ in range(self.depth - 1):
            supports_layers.extend([nn.ReLU(), nn.Linear(self.width, self.width)])
            params_layers.extend([nn.ReLU(), nn.Linear(self.width, self.width)])

        self.supports_trunk = nn.Sequential(*supports_layers)
        self.params_trunk = nn.Sequential(*params_layers)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.width,
            num_heads=num_heads,

        )

    def forward(self, supports_embeddings: torch.Tensor, params: torch.Tensor):
        support_emb = self.supports_trunk(supports_embeddings)
        params_emb = self.params_trunk(params)

        _, attn_mask = self.attention(
            query=support_emb,
            key=params_emb,
            value=torch.zeros_like(params_emb)
        )

        return attn_mask


class InvariantMAMLAttention(nn.Module):

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

        self.ranker = RankerAttention(args, hdim, )

        self.permutation_to_idx = OrderedDict()
        self.idx_to_permutation = OrderedDict()

        for i, p in enumerate(itertools.permutations(list(range(args.way)))):
            self.permutation_to_idx[p] = i
            self.idx_to_permutation[i] = p

    def forward(self, data_shot, data_query, permutation):
        # update with gradient descent
        updated_params = inner_train_step(model=self.encoder,
                                          support_data=data_shot,
                                          args=self.args,
                                          permutation=permutation)

        logits = self.encoder(data_query, updated_params) / self.args.temperature
        return logits

    def train_ranker(self, support_data, best_permutation, args):
        avg_enhanced_embeddings = get_enhanced_embeddings(self.encoder, support_data, args)

        metrics = {}

        scores = self.ranker(
            avg_enhanced_embeddings.detach(), self.encoder.fc.weight.detach()
        )

        loss = F.cross_entropy(
            input=scores,
            target=torch.tensor(best_permutation).cuda()
        )

        # multiply scores by -1 for LAP, because LAP aims to find lowest cost
        rids, cids = lap_solve_dense(-scores.detach().cpu().numpy())

        predicted_permutation = tuple(cids)

        metrics["permutation"] = 1 if predicted_permutation == best_permutation else 0
        for i in range(args.way):
            metrics[f"ranker_accuracy{i}"] = 1 if predicted_permutation[i] == best_permutation[i] else 0

        return loss, metrics

    def forward_eval(self, data_shot, data_query, args):
        # update with gradient descent
        self.train()
        avg_enhanced_embeddings = get_enhanced_embeddings(self.encoder, data_shot, args)
        scores = self.ranker(
            avg_enhanced_embeddings.detach(), self.encoder.fc.weight.detach()
        )

        # multiply scores by -1 for LAP, because LAP aims to find lowest cost
        rids, cids = lap_solve_dense(-scores.detach().cpu().numpy())

        predicted_permutation = tuple(cids)

        updated_params = inner_train_step(model=self.encoder,
                                          support_data=data_shot,
                                          args=self.args,
                                          permutation=predicted_permutation)

        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature

        return logitis_query, predicted_permutation
