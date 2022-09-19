import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler, ClassSampler
from collections import Counter



def get_dataloader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CIFARFS':
        from model.dataloader.cifarfs import CIFARFS as Dataset
        args.dropblock_size = 2                
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import FC100 as Dataset      
        args.dropblock_size = 2        
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet_raw import tieredImageNet as Dataset    
        args.dropblock_size = 5        
    else:
        raise ValueError('Non-supported Dataset.')

    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch
    num_workers = args.num_workers * num_device
    
    trainset = Dataset('train', args)
    args.num_class = trainset.num_class
    train_sampler = CategoriesSampler(trainset.label,
                                      num_episodes, args.way,
                                      args.shot + args.query)
    train_loader = DataLoader(dataset=trainset,
                              num_workers=num_workers,
                              batch_sampler=train_sampler,
                              pin_memory=True)
    
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label,
                                    args.num_eval_episodes, args.eval_way,
                                    args.eval_shot + args.eval_query)
    val_loader = DataLoader(dataset=valset,
                            batch_sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)
    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                                     args.num_test_episodes, args.eval_way,
                                     args.eval_shot + args.eval_query)
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)        

    return train_loader, val_loader, test_loader

def get_cross_shot_dataloader(args, shot):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CIFARFS':
        from model.dataloader.cifarfs import CIFARFS as Dataset
        args.dropblock_size = 2                
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import FC100 as Dataset      
        args.dropblock_size = 2        
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet_raw import tieredImageNet as Dataset    
        args.dropblock_size = 5                
    else:
        raise ValueError('Non-supported Dataset.')
    
    testset = Dataset('test', args)
    test_sampler = CategoriesSampler(testset.label,
                                     args.num_test_episodes,
                                     args.eval_way,
                                     shot + args.eval_query)    
    test_loader = DataLoader(dataset=testset,
                            batch_sampler=test_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True)        

    return test_loader

def get_class_dataloader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
        args.dropblock_size = 5        
    elif args.dataset == 'CIFARFS':
        from model.dataloader.cifarfs import CIFARFS as Dataset
        args.dropblock_size = 2                
    elif args.dataset == 'FC100':
        from model.dataloader.fc100 import FC100 as Dataset      
        args.dropblock_size = 2        
    elif args.dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet_raw import tieredImageNet as Dataset    
        args.dropblock_size = 5                
    else:
        raise ValueError('Non-supported Dataset.')

    trainset = Dataset('train', args) # do not augment when extracting features
    args.num_class = trainset.num_class
    class_sampler = ClassSampler(trainset.label,  
                                 min(Counter(trainset.label).values()), # 10
                                 ) # at least 30 items per item in the queue
    class_loader = DataLoader(dataset=trainset,
                             num_workers=args.num_workers,
                             batch_sampler=class_sampler,
                             pin_memory=True)     

    return class_loader

def prepare_model(args):
    if args.model_class == 'MAML':
        from model.models.maml import MAML 
        model = MAML(args)
    elif args.model_class == 'MAMLUnicorn':
        from model.models.MAMLUnicorn import MAML 
        model = MAML(args)
    elif args.model_class == "HyperMAML":
        from model.models.hypermaml import HyperMAML
        model = HyperMAML(args)
    else:
        raise ValueError('No Such Model')
    
    # load pre-trained model (no FC weights)
    if args.para_init is not None:
        model_dict = model.state_dict()        
        pretrained_dict = torch.load(args.para_init, map_location='cpu')['params'] # map_location=torch.device('cpu')

        if args.backbone_class == "Conv4":
            pd2 = dict()
            for k, v in pretrained_dict.items():
                if k.startswith("encoder"):
                    p1, p2, p3, p4 = k.split(".")
                    new_k = f"{p1}.{p2}_{p3}.{p4}"
                    pd2[new_k] = v
                else:
                    pd2[k] = v

            pretrained_dict = pd2

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if len(args.gpu.split(',')) > 1:
        model = nn.DataParallel(model, dim=0).to(device)
    else:
        model = model.to(device)

    return model

def prepare_optimizer(model, args):

    # select parameters
    if args.lr_mul > 1:
        top_para = [v for k,v in model.named_parameters() if 'encoder' not in k]

        params = [{'params': model.encoder.parameters()},
                 {'params': top_para, 'lr': args.lr * args.lr_mul}]

        if args.um_freeze_backbone:
            from model.models.hypermaml import HyperMAML

            assert isinstance(model, HyperMAML)
            params = model.hn.parameters()



    else:
        params =model.parameters()

    # select optimizer
    if args.optimizer_class == "sgd":

        optimizer = optim.SGD(params, lr=args.lr,
                         momentum=args.mom,
                         nesterov=True,
                         weight_decay=args.weight_decay)

    elif args.optimizer_class == "adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    else:
        raise NotImplementedError(args.optimizer_class)

    # select lr_scheduler
    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=int(args.step_size),
                            gamma=args.gamma
                        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=[int(_) for _ in args.step_size.split(',')],
                            gamma=args.gamma,
                        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            args.max_epoch,
                            eta_min=0   # a tuning parameter
                        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
