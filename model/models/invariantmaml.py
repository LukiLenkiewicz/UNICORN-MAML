import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import itertools
from collections import OrderedDict
from model.utils import count_acc

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

    for l in set(label.cpu().numpy()):
        avg_enhanced_embeddings.append(
            enhanced_embeddings[label==l].mean(dim=0)
        )

    avg_enhanced_embeddings = torch.stack(avg_enhanced_embeddings)
    return avg_enhanced_embeddings

def inner_train_step(model, support_data, ranker, permutations, args):
    """ Inner training step procedure. """

    best_permutation = None
    best_params = None
    max_acc = 0.0

    for base_permutation in permutations:
        # obtain final prediction
        updated_params = OrderedDict(model.named_parameters())
        label = torch.tensor(base_permutation).repeat(args.shot)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)

        for _ in range(args.inner_iters):
            ypred = model(support_data, updated_params)
            loss = F.cross_entropy(ypred, label)
            updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)

        crr_acc = count_acc(ypred, label)

        if crr_acc >= max_acc:
            max_acc = crr_acc
            best_permutation = base_permutation
            best_params = updated_params

    avg_enhanced_embeddings = get_enhanced_embeddings(model, support_data, args)
    pred_perm_ranking_scores = ranker(avg_enhanced_embeddings.view(1, -1))

    return best_params, pred_perm_ranking_scores, best_permutation

def inner_test_step(model, support_data, ranker, idx_to_permutation, args):
    """ Inner testing step procedure. """
    
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    avg_enhanced_embeddings = get_enhanced_embeddings(model, support_data, args)

    # predict permutation
    pred_perm_ranking_scores = ranker(avg_enhanced_embeddings.view(1, -1))
    predicted_permutation = idx_to_permutation[pred_perm_ranking_scores.argmax().item()]
    
    predicted_labels = torch.tensor(predicted_permutation).repeat(args.shot)

    if torch.cuda.is_available():
        predicted_labels = predicted_labels.type(torch.cuda.LongTensor)
    else:
        predicted_labels = predicted_labels.type(torch.LongTensor)         

    for _ in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.cross_entropy(ypred, predicted_labels)
        updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)
    
    return updated_params, pred_perm_ranking_scores, predicted_permutation

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
        
        self.ranker = Ranker(args, hdim)

        self.permutation_to_idx = OrderedDict()
        self.idx_to_permutation = OrderedDict()

        for i, p in enumerate(itertools.permutations(list(range(args.way)))):
            self.permutation_to_idx[p] = i
            self.idx_to_permutation[i] = p

    def forward(self, data_shot, data_query):
        # update with gradient descent
        updated_params, pred_perm_ranking_scores, best_permutation = inner_train_step(model=self.encoder,
                                                                                      support_data=data_shot,
                                                                                      ranker=self.ranker,
                                                                                      permutations=self.permutation_to_idx.keys(),
                                                                                      args=self.args)

        predicted_permutation = self.idx_to_permutation[pred_perm_ranking_scores.argmax(dim=1).item()]

        if self.args.predicted_perm_train:
            result_permutation = predicted_permutation
        else:
            result_permutation = best_permutation

        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis, pred_perm_ranking_scores, result_permutation

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

    def forward_eval(self, data_shot, data_query):
        # update with gradient descent
        self.train()
        updated_params, pred_perm_ranking_scores, labels_permutation = inner_test_step(model=self.encoder,
                                                                                       support_data=data_shot,
                                                                                       ranker=self.ranker,
                                                                                       idx_to_permutation=self.idx_to_permutation,
                                                                                       args=self.args)

        # get shot accuracy and loss
        self.eval()
        with torch.no_grad():
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis_query, pred_perm_ranking_scores, labels_permutation


class Ranker(nn.Module):
    def __init__(self, args, hdim, last_activation_fn='sigmoid'):
        super(Ranker, self).__init__()

        self.hdim = hdim
        self.depth = args.ranker_depth
        self.width = args.ranker_width
        self.in_neurons = (self.hdim + 2 * args.way) * args.way
        self.out_neurons = math.factorial(args.way)

        layers = []
        
        for i in range(self.depth):
            in_neurons_num = self.in_neurons if i == 0 else self.width
            out_neurons_num = self.out_neurons if i == self.depth - 1 else self.width
            
            layers.append(nn.Linear(in_neurons_num, out_neurons_num))
            
            if i < self.depth - 1:
                layers.append(nn.ReLU())

        if last_activation_fn == 'relu':
            layers.append(nn.ReLU())
        elif last_activation_fn == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

