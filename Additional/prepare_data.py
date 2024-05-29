import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset, ConcatDataset
from distributed_utils import is_main_process




def get_loader(args, train, seed=0, download=False):
    if is_main_process():
        print('==> Preparing data..\n')
    if train:
        if args.aug:
            print("Loading training data with augmentation")
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        else:
            print("Loading training data without augmentation")
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        dataset = torchvision.datasets.CIFAR10(root=args.data_pth, train=True, download=download, transform=transform_train)
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
        dataset = torchvision.datasets.CIFAR10(root=args.data_pth, train=False, download=download, transform=transform_test)

    shuffle = None
    if train:
        if args.replacement:
            num_samples = args.batch_size_per_gpu * args.steps_per_epoch
            sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)

            if args.debug and is_main_process():
                print(f"num samples per epoch per gpu: {num_samples}")
        # elif not(args.original):
            # print("Fixed local datasets")
            # N = 50000
            # np.random.seed(seed)
            # subsets = [Subset(dataset, 
            #                 np.random.permutation(np.arange(i*N // args.num_clients, (i+1)*N // args.num_clients))) for i in range(args.num_clients)]
            # dataset = ConcatDataset(subsets)

            # if args.distributed:
            #     sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
            # else:
            #     sampler = None
        else:
            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, seed=seed)
            else:
                sampler = None
                shuffle = True
    else:
        sampler = None

    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, sampler=sampler, shuffle=shuffle, num_workers=args.nw, drop_last=train)
    
    
    return dataloader, sampler
