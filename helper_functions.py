import os
from pathlib import Path
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
import random
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def walk_through_dir(dir_path):
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} audios in {dirpath}")


def find_classes(directory: Path):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    if not classes:
        raise FileNotFoundError("Couldn't find any classes, please check file structure")
    class_to_idx = {class_name:i for i, class_name in enumerate(classes)}
    return classes, class_to_idx


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device=device):
    
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() # maybe should add .item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class==y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)    
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device=device):
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(test_pred_logits, dim=1)
            test_acc += (test_pred_labels==y).sum().item() / len(test_pred_logits)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc 


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,          
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          scheduler: torch.optim.lr_scheduler = None,
          epochs: int = 5,
          device=device):
    
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        if scheduler:
             scheduler.step(test_loss)

        print(f"Epoch: {epoch} | Train loss {train_loss:.4f}, Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
    return results



def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_acc = results["test_acc"]

    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15,7))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title("Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_acc, label='test_accuracy')
    plt.title('Accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel('Epochs')
    plt.legend()


def make_predictions(model: nn.Module,
                     data: Dataset,
                     class_names: List,
                     n: int = 10,
                     device=device):                   

        count_correct = 0
        rand_idxs = random.sample(range(len(data)), k=n)
        model.eval()
        with torch.inference_mode():
                for idx in rand_idxs:
                        y_pred = model(data[idx][0].unsqueeze(dim=0).to(device))
                        pred_label = torch.argmax(y_pred, dim=1)
                        print(f"Prediction genre: {class_names[pred_label]} | True genre: {class_names[data[idx][1]]}")
                        count_correct += pred_label==data[idx][1]
        print("-----------") 
        print(f"Correct preditions: {count_correct.item()} out of {n}\n-----------")       


def plot_confmat(model: nn.Module,
                 test_data: Dataset,
                 class_names: List,
                 device=device):
    y_preds = []
    y_targets = []

    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_data):
            X = X.unsqueeze(dim=0).to(device)
            y_logits_tensor = model(X)
            y_pred = torch.argmax(y_logits_tensor, dim=1)
            y_preds.append(y_pred.cpu())
            y_targets.append(torch.tensor([y]))
        
    y_pred_tensor = torch.cat(y_preds)
    y_targets_tensor = torch.cat(y_targets)
    
    confmat = ConfusionMatrix(num_classes=len(test_data.classes), task='multiclass')
    confmat_tensor = confmat(preds=y_pred_tensor, target=y_targets_tensor)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=class_names,
        figsize=(10,7)
    )
