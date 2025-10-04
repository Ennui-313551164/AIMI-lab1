import os
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder

import seaborn as sns
from matplotlib.ticker import MaxNLocator 

import logging


def measurement(outputs, labels, smooth=1e-10):
    tp, tn, fp, fn = smooth, smooth, smooth, smooth
    labels = labels.cpu().numpy()
    outputs = outputs.detach().cpu().clone().numpy()
    for j in range(labels.shape[0]):
        if (int(outputs[j]) == 1 and int(labels[j]) == 1):
            tp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 0):
            tn += 1
        if (int(outputs[j]) == 1 and int(labels[j]) == 0):
            fp += 1
        if (int(outputs[j]) == 0 and int(labels[j]) == 1):
            fn += 1
    return tp, tn, fp, fn


def plot_accuracy(train_acc_list, val_acc_list, mode):
    # Plot training accuracy curve and final validation accuracy
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_acc_list) + 1)
    
    # Plot training accuracy curve and validation accuracy curve
    plt.plot(epochs, train_acc_list, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    plt.plot(epochs, val_acc_list, 'r-', label='Validation Accuracy', linewidth=2, marker='o', markersize=4)
    
    
    plt.title('Training Accuracy and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set integer ticks for x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set y-axis limits with some padding
    all_acc = train_acc_list + val_acc_list
    y_min = max(0, min(all_acc) - 5)
    y_max = min(100, max(all_acc) + 5)
    plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    # plt.savefig('accuracy_curve.png', dpi=300, bbox_inches='tight')

    if mode == 'train':
        plt.savefig(f"result/plot/{model_name}_accuracy_curve.png")
    elif mode == 'test':
        plt.savefig(f"result/plot/{model_name}_test_accuracy_curve.png")
    # plt.show()


def plot_f1_score(f1_score_list, model_name, mode):
    # Plot F1 score curve over epochs
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(f1_score_list) + 1)
    
    plt.plot(epochs, f1_score_list, 'g-', label='F1 Score', linewidth=2, marker='D', markersize=5)
    
    plt.title('F1 Score Over Epochs', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set integer ticks for x-axis
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Set y-axis limits (F1 score is between 0 and 1)
    y_min = max(0, min(f1_score_list) - 0.05)
    y_max = min(1, max(f1_score_list) + 0.05)
    plt.ylim(y_min, y_max)
    
    # Add horizontal lines for reference
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Baseline (0.5)')
    plt.axhline(y=0.8, color='orange', linestyle=':', alpha=0.5, label='Good Performance (0.8)')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    # plt.savefig('f1_score_curve.png', dpi=300, bbox_inches='tight')
    if mode == 'train':
        plt.savefig(f"result/plot/{model_name}_f1_score.png")
    elif mode == 'test':
        plt.savefig(f"result/plot/{model_name}_test_f1_score.png")
    # plt.show()


def plot_confusion_matrix(confusion_matrix, model_name, mode):
    # TODO plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Normal', 'Predicted Pneumonia'],
                yticklabels=['Actual Normal', 'Actual Pneumonia'])
    plt.title(f'Confusion Matrix {model_name}')

    if mode == 'train':
        plt.savefig(f"result/plot/{model_name}_confusion.png")
    elif mode == 'test':
        plt.savefig(f"result/plot/{model_name}_test_confusion.png")


def train(device, train_loader, val_loader, model, criterion, optimizer, model_name):
    best_acc = 0.0
    best_model_wts = None
    train_acc_list = []
    val_acc_list = []
    f1_score_list = []
    best_c_matrix = []

    for epoch in range(1, args.num_epochs+1):
        model.train()

        with torch.set_grad_enabled(True):
            avg_loss = 0.0
            train_acc = 0.0
            tp, tn, fp, fn = 0, 0, 0, 0     
            for _, data in enumerate(tqdm(train_loader)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                outputs = torch.max(outputs, 1).indices
                sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
                tp += sub_tp
                tn += sub_tn
                fp += sub_fp
                fn += sub_fn          

            avg_loss /= len(train_loader.dataset)
            train_acc = (tp+tn) / (tp+tn+fp+fn) * 100
            print(f'Epoch: {epoch}')
            print(f'↳ Loss: {avg_loss}')
            print(f'↳ Training Acc.(%): {train_acc:.2f}%')
            logging.info(f'Epoch: {epoch}')
            logging.info(f'↳ Loss: {avg_loss}')
            logging.info(f'↳ Training Acc.(%): {train_acc:.2f}%')
            
        #?
        torch.save(model.state_dict(), f'result/{model_name}/weight{epoch}.pt')  # save model weights each epoch

        # write validation if you needed
        val_acc, f1_score, c_matrix = test(val_loader, model)

        if val_acc > best_acc:
            best_acc = val_acc
            best_c_matrix = c_matrix
            best_model_wts = model.state_dict()

        
        train_acc_list.append(train_acc)
        #?
        val_acc_list.append(val_acc)
        f1_score_list.append(f1_score)

    torch.save(best_model_wts, f'result/{model_name}_best_model.pt')  # save best model weights

    return train_acc_list, val_acc_list, f1_score_list, best_c_matrix


def test(test_loader, model):
    val_acc = 0.0
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.set_grad_enabled(False):
        model.eval()
        for images, labels in test_loader:
            
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = torch.max(outputs, 1).indices

            sub_tp, sub_tn, sub_fp, sub_fn = measurement(outputs, labels)
            tp += sub_tp
            tn += sub_tn
            fp += sub_fp
            fn += sub_fn

        c_matrix = [[int(tp), int(fn)],
                    [int(fp), int(tn)]]
        
        val_acc = (tp+tn) / (tp+tn+fp+fn) * 100
        recall = tp / (tp+fn)
        precision = tp / (tp+fp)
        f1_score = (2*tp) / (2*tp+fp+fn)
        print (f'↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}')
        print (f'↳ Test Acc.(%): {val_acc:.2f}%')
        logging.info(f"↳ Recall: {recall:.4f}, Precision: {precision:.4f}, F1-score: {f1_score:.4f}")
        logging.info(f"↳ Test Acc.(%): {val_acc:.2f}")

    return val_acc, f1_score, c_matrix



if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    parser = ArgumentParser()

    # for model
    parser.add_argument('--num_classes', type=int, required=False, default=2)
    parser.add_argument('--model', type=str, required=False)

    # for training
    parser.add_argument('--num_epochs', type=int, required=False, default=20)
    parser.add_argument('--batch_size', type=int, required=False, default=64)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.9)

    # for dataloader (dataset path)
    parser.add_argument('--dataset', type=str, required=False, default='chest_xray')

    # for data augmentation
    parser.add_argument('--degree', type=int, default=90)
    parser.add_argument('--resize', type=int, default=224)

    args = parser.parse_args()


    ### ========================== Parameter setting ========================== ###
    model_name = args.model  # TODO: change the model name by yourself
    DATASET_PATH = args.dataset

    SAVE_PATH = f'result/{model_name}'
    mode = 'train'  # 'train' or 'test'


    ### ========================== Parameter setting ========================== ###
    #? Create visualization directory if not exists
    if not os.path.exists(f'result/{model_name}'):
        os.makedirs(f'result/{model_name}')
    
    if not os.path.exists(f'result/plot'):
        os.makedirs(f'result/plot')

    #? Set up logging to file
    logging.basicConfig(
        filename='train_log.txt',
        # filemode='w',  # overwrite the log file each time
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # set gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'## Now using {device} as calculating device ##')

    # set dataloader (Train and Test dataset, write your own validation dataloader if needed.)
    # TODO / Change the data augmentation method yourself 
    train_dataset = ImageFolder(root=os.path.join(DATASET_PATH, 'train'),
                                transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                                # transforms.RandomRotation(args.degree, resample=False),
                                                                transforms.RandomRotation(args.degree),
                                                                transforms.ToTensor()]))
    
    #?
    val_dataset = ImageFolder(root=os.path.join(DATASET_PATH, 'val'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))

    test_dataset = ImageFolder(root=os.path.join(DATASET_PATH, 'test'),
                               transform = transforms.Compose([transforms.Resize((args.resize, args.resize)),
                                                               transforms.ToTensor()]))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #?
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # TODO / define model 
    if model_name == 'ResNet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'ResNet101':
        model = models.resnet101(pretrained=True)
    else:
        raise ValueError('Invalid model name, please check the model name again.')

    # 原來: model.fc 輸出 1000 個類別
    # 現在: model.fc 輸出 2 個類別 (Normal, Pneumonia)
    num_neurons = model.fc.in_features 
    model.fc = nn.Linear(num_neurons, args.num_classes)
    model = model.to(device)


    ### ========================= Model train/val ========================== ###

    # define loss function, optimizer
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([3.8896346, 1.346]))
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


    # training
    train_acc_list, val_acc_list, f1_score_list, _ = train(device, train_loader, val_loader, model, criterion, optimizer, model_name)

    ### Plotting Training/Validation Accuracy, F1 Score, Confusion Matrix ###
    plot_accuracy(train_acc_list, val_acc_list, mode)
    plot_f1_score(f1_score_list, model_name, mode)
    # plot_confusion_matrix(best_c_matrix, model_name, mode)

    ### ========================= Model train/val ========================== ###


    ### ========================== Model testing ========================== ###
    print("========== Test dataset for the best model ==========")
    logging.info("========== Test dataset for the best model ==========")
    mode = 'test' #?

    # 1. load best model weights
    state_dict = torch.load(f'result/{model_name}_best_model.pt', map_location=device)
    model.load_state_dict(state_dict)

    # 2. test
    test_acc, f1_score, best_c_matrix= test(test_loader, model)

    ### Plotting Training/Validation Accuracy, F1 Score, Confusion Matrix ###
    # plot_accuracy(train_acc_list, val_acc_list, mode)
    # plot_f1_score(f1_score_list, model_name, mode)
    plot_confusion_matrix(best_c_matrix, model_name, mode)

    ### ========================== Model testing ========================== ###