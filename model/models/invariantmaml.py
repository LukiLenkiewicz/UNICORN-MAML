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

def update_params(loss, params, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad

    return updated_params

def get_enhanced_embeddings(model, support_data, args, permutation=None):
    if permutation is None:
        label = torch.arange(args.way).repeat(args.shot)
    else:
        label = torch.tensor(permutation).repeat(args.shot)

    params = OrderedDict(model.named_parameters())
    embeddings, logits = model(support_data, params, embedding_and_logits=True)
    
    if args.ranker_input_pooling not in ['average', 'max', 'min']:
        onehot = torch.zeros(len(support_data), args.way)
        onehot[torch.arange(len(support_data)), label] = 1
        
        enhanced_embeddings = torch.cat([embeddings, logits, onehot.cuda()], dim=1)
    else:
        enhanced_embeddings = embeddings

    avg_enhanced_embeddings = []

    for l in set(label.cpu().numpy()):
        avg_enhanced_embeddings.append(
            enhanced_embeddings[label==l].mean(dim=0)
        )

    avg_enhanced_embeddings = torch.stack(avg_enhanced_embeddings)

    if args.ranker_input_pooling == 'average':
        avg_enhanced_embeddings = torch.mean(avg_enhanced_embeddings, dim=0)
    elif args.ranker_input_pooling == 'max':
        avg_enhanced_embeddings = torch.max(avg_enhanced_embeddings, dim=0).values
    else:
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

class InvariantMAML(nn.Module):

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
        
        self.ranker = Ranker(args, hdim, math.factorial(args.way))

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

        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis

    def train_ranker(self, support_data, best_permutation, args):
        avg_enhanced_embeddings = get_enhanced_embeddings(self.encoder, support_data, args)
        scores = self.ranker(avg_enhanced_embeddings.view(1, -1)).flatten()
        predicted_permutation_idx = scores.argmax(dim=0).item()
        best_permutation_idx = self.permutation_to_idx[best_permutation]
        
        predicted_permutation = self.idx_to_permutation[predicted_permutation_idx]
        
        metrics = { "permutation": 1 if predicted_permutation_idx == best_permutation_idx else 0 }
        for i in range(args.way):
            metrics[f"permutation_pos{i}"] = 1 if predicted_permutation[i] == best_permutation[i] else 0

        expected = torch.tensor(best_permutation_idx).cuda()
        return F.cross_entropy(scores, expected), metrics

    def forward_eval_perm(self, data_shot, data_query):
        # update with gradient descent
        # for permutation evaluation, and will output some statistics
        original_params = OrderedDict(self.named_parameters())
        self.train()
        updated_params = inner_train_step(self.encoder, data_shot, self.args)
        
        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_shot = self.encoder(data_shot, updated_params)
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
            
        # compute the norm of the params
        norm_list =  [torch.norm(updated_params[e] - original_params['encoder.' + e]).item() for e in updated_params.keys() ]
        return logitis_shot, logitis_query, np.array(norm_list) 
    
    
    def forward_eval_ensemble(self, data_shot, data_query_list):
        # update with gradient descent for Ensemble evaluation
        self.train()
        updated_params = inner_train_step(self.encoder, data_shot, self.args)
        
        # get shot accuracy and loss
        self.eval()
        logitis_query_list = []
        with torch.no_grad():
            for data_query in data_query_list:
                # logitis_shot = self.encoder(data_shot, updated_params)
                logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
                logitis_query_list.append(logitis_query)
        return logitis_query_list # logitis_shot,     

    def forward_eval(self, data_shot, data_query, args, permutation=None):
        # update with gradient descent
        self.train()
        avg_enhanced_embeddings = get_enhanced_embeddings(self.encoder, data_shot, args, permutation)
        
        scores = self.ranker(avg_enhanced_embeddings.view(1, -1)).flatten()
        
        if permutation is None:
            predicted_permutation_idx = scores.argmax(dim=0).item()
            predicted_permutation = self.idx_to_permutation[predicted_permutation_idx]
        else:
            predicted_permutation = permutation
        
        updated_params = inner_train_step(model=self.encoder,
                                          support_data=data_shot,
                                          args=self.args,
                                          permutation=predicted_permutation)

        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature

        return logitis_query, predicted_permutation