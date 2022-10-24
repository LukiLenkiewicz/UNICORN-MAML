import torch
import torch.nn as nn
import math

class Ranker(nn.Module):
    def __init__(self, args, hdim, out_neurons, last_activation_fn='sigmoid'):
        super(Ranker, self).__init__()

        self.args = args
        self.hdim = hdim
        self.depth = args.ranker_depth
        self.width = args.ranker_width
        self.out_neurons = out_neurons
        
        if args.feed_heads_with_support_embeddings:
            self.in_neurons = (self.hdim + 2 * args.way) * args.way
        else:
            self.in_neurons = (self.hdim + 2 * args.way)
            
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

