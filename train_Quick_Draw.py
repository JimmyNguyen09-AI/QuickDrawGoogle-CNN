# author : Trung Thanh Nguyen(Jimmy) | 09/12/2004  | ng.trungthanh04@gmail.com
import os.path
import numpy as np
import torch
from quickDrawDataset import QuickDrawDataset
from model_CNN_QuickDraw import CNN
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score,confusion_matrix
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import shutil
import matplotlib.pyplot as plt
def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)
def get_args():
    parser = argparse.ArgumentParser("Test Argument")
    parser.add_argument("--image-size", "-i", type=int, default=28)
    parser.add_argument("--batch-size","-b",type = int,default=32)
    parser.add_argument("--epochs","-e",type = int,default=50)
    parser.add_argument("--learning-rate","-l",type = float,default = 1e-2)
    parser.add_argument("--log-path","-p",type = str,default="tensor_board_quickdraw/quickdraw")
    parser.add_argument("--momentum","-m",type = float,default = 0.9)
    parser.add_argument("--data-path","-d",type = str,default = "./dataset_Quick_Draw")
    parser.add_argument("--checkpoint-path", "-c", type=str, default="train_model_QuickDraw")
    parser.add_argument("--pretrained-path", "-t", type=str, default=None)
    parser.add_argument("--es-patience", type=int, default=3)
    parser.add_argument("--es-min-delta", type=float, default=0.001)
    args = parser.parse_args()
    return args
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = QuickDrawDataset(root = args.data_path,is_train=True,nums_images_per_class = 10000,ratio = 0.8)
    train_params = {
        "batch_size": args.batch_size,
        "shuffle":True,
        "num_workers" :4,
        "drop_last": True
    }
    train_dataloader = DataLoader(dataset=train_dataset, **train_params)
    num_iter_per_epoch = len(train_dataloader)
    val_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 4,
        "drop_last": False
    }
    val_dataset = QuickDrawDataset(root = args.data_path,is_train=False,nums_images_per_class = 10000,ratio = 0.8)
    val_dataloader = DataLoader(dataset=val_dataset, **val_params)
    model = CNN(num_classes = len(train_dataset.categories)).to(device)
    # model = resnet18(weights = ResNet18_Weights.DEFAULT)
    # model.fc = nn.Linear(512, 10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = args.learning_rate,momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    if not os.path.isdir(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)
    best_loss = 1e6
    best_epoch = 0
    model.train()
    for epoch in range(args.epochs):
        train_loss = []
        progress_bar = tqdm(train_dataloader,colour = "cyan")
        for iter,(images,labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions,labels)
#           backward voi optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            progress_bar.set_description("Epoch: {}/{}. Loss: {:0.4f}".format(epoch+1,args.epochs,loss))
            writer.add_scalar("Train/Loss",loss,epoch*num_iter_per_epoch + iter)
        scheduler.step()
        # model validation
        val_losses = []
        all_predictions = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, colour="yellow")
            for iter, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)
                loss = criterion(predictions, labels)
                predictions = torch.argmax(predictions, 1)
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())
                val_losses.append(loss.item())
        acc = accuracy_score(all_labels,all_predictions)
        te_loss = np.mean(val_losses)
        print("Accuracy: {} Loss: {} ".format(acc,te_loss))
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        writer.add_scalar("Validation/Loss", te_loss, epoch)
        writer.add_scalar("Validation/Accuracy", acc, epoch)
        plot_confusion_matrix(writer, conf_matrix, [i for i in range(10)], epoch)
        model.train()
        if te_loss + args.es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            model_save_path = os.path.join(args.checkpoint_path, "quickdraw")
            torch.save(model, model_save_path)
            print(f"New best model saved at epoch {epoch + 1} to {model_save_path}")
        if epoch - best_epoch > args.es_patience > 0:
            print(f"Stopping training at epoch {epoch + 1}. Best validation loss: {best_loss:.4f}")
            break
    writer.close()

if __name__ == '__main__':
    args = get_args()
    train(args)