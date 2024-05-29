import glob
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import pandas as pd
from models import resnets, vggnets
from prepare_data import get_loader
from train_utils import count_correct




def inference(args):
    args.original = True
    args.distributed = False
    args.nw = 1
    args.aug = True
    args.seed = 0
    device = torch.device(args.device)

    if 'resnet' in args.model:
        model = getattr(resnets, args.model)().to(device)
    else:
        raise NotImplementedError()
    
    model.load_state_dict(torch.load(args.model_pth, map_location=device))
    
    trainloader, _ = get_loader(args, train=True)
    testloader, _ = get_loader(args, train=False)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    # Train set inference
    train_loss, total, correct = 0.0, 0.0, 0.0
    with autocast():
        for images, targets in trainloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            train_loss += criterion(outputs, targets).item()
            correct += count_correct(
                output=outputs,
                target=targets,
                topk=(1,)
            )[0].item()
            total += len(targets)
    train_accuracy = correct / total
    train_loss /= len(trainloader)

    # Test set inference
    test_loss, total_, correct_ = 0.0, 0.0, 0.0
    with autocast():
        for images, targets in testloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, targets).item()
            correct_ += count_correct(
                output=outputs,
                target=targets,
                topk=(1,)
            )[0].item()
            total_ += len(targets)
    test_accuracy = correct_ / total_
    test_loss /= len(testloader)


    return (train_loss, test_loss, train_accuracy, test_accuracy)


def main(args):
    df = pd.DataFrame(columns=["train_loss", "test_loss", "train_acc", "test_acc"])
    filename = args.save_pth+'/values_R'
    for file in glob.glob(args.save_pth+'/*.pt'): 
        args.model_pth = str(file)
        R = int(args.model_pth.replace(args.save_pth+'/latest_R', '').replace('.pt', ''))
        df.loc[R] = inference(args)
        filename += str(R)+'_'
    filename += ".pickle"
    df = df.sort_index()
    df.to_pickle(filename)



if __name__ == '__main__':
    model_names_resnet = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
    model_names_vgg = ['vgg11', 'vgg13', 'vgg16', 'vgg19']
    model_names = [name + '_gn' for name in model_names_resnet]
    model_names.extend(model_names_resnet)
    model_names.extend(model_names_vgg)
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-pth', type=str, default='./data', help='path to data')
    parser.add_argument('--save-pth', type=str, help='path to folder for saving')
    parser.add_argument('--model', type=str, default='resnet56', choices=model_names, help='The model to use')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for inference')
    parser.add_argument('--replacement', type=int, default=0, help='whether to use sampling with replacement')

    args = parser.parse_args()

    main(args)
    # args.model_pth = "./checkpoint/test1/latest_R2.pt"
    # print(inference(args))