import os
import time
import numpy as np
import csv
import torch
import torch.utils.data as data
from utils.loss_function import DiceBCELoss, BayesBCELoss
from models.bayesian_deeplab import DeepLabv3plusModel
from models.bayesian_unet import Unet256
from utils.module import EarlyStopping, write_to_csv
from dataloader.dataset import make_datapath_list, LabeledDataset, LabeledTransform, ValLabeledTransform, UnlabeledDataset, UnlabeledTransform


def train(former_folname, folname, net="deeplab", batch_size=16, num_workers=2, epochs=500, alpha=1000, beta=10):
    # make dataloader
    makepath = make_datapath_list(former_folname)
    train_labeled_img_list, train_labeled_anno_list = makepath.get_list("train_labeled")
    train_unlabeled_img_list, train_unlabeled_mean_list, train_unlabeled_var_list = makepath.get_list("train_unlabeled")
    val_img_list, val_anno_list = makepath.get_list("val")

    train_labeled_dataset = LabeledDataset(train_labeled_img_list, train_labeled_anno_list, transform=LabeledTransform(crop_size=256))
    val_dataset = LabeledDataset(val_img_list, val_anno_list, transform=ValLabeledTransform(crop_size=256))
    train_unlabeled_dataset = UnlabeledDataset(train_unlabeled_img_list, train_unlabeled_mean_list, train_unlabeled_var_list, transform=UnlabeledTransform(crop_size=256, rotation=True))

    train_labeled_dataloader = data.DataLoader(
        train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    train_unlabeled_dataloader = data.DataLoader(
        train_unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # define loss function
    class_criterion = DiceBCELoss()
    cons_criterion = BayesBCELoss(alpha,beta)

    # define model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if net == "deeplab":
        model_wrapper = DeepLabv3plusModel(device)
        model = model_wrapper.get_model()
    else:
        model = Unet256((3, 256, 256)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    start_epoch = 0
    earlystopping = EarlyStopping(patience=50)

    os.makedirs('weights/'+ folname +'_weights', exist_ok=True)
    os.makedirs('loss/'+ folname +'_loss', exist_ok=True)

    csv_filename = 'loss/'+folname+'_loss/loss.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Consistency Loss', 'Validation Loss'])

    model = model.to(device)

    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()

        running_train_loss = []
        running_train_cons_loss = []
        model.train()
        for image,mask in train_labeled_dataloader:
            image = image.to(device,dtype=torch.float)
            mask = mask.to(device,dtype=torch.float)
            pred_mask = model.forward(image)
            loss = class_criterion(pred_mask,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())

        for image, mean, var in train_unlabeled_dataloader:
            image = image.to(device,dtype=torch.float)
            mean = mean.to(device,dtype=torch.float)
            var = var.to(device,dtype=torch.float)

            pred_mask = model.forward(image)
            loss = cons_criterion(pred_mask,mean,var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_cons_loss.append(loss.item())

        running_val_loss = []
        model.eval()
        with torch.no_grad():
            for image,mask in val_dataloader:
                image = image.to(device,dtype=torch.float)
                mask = mask.to(device,dtype=torch.float)
                pred_mask = model.forward(image)
                loss = class_criterion(pred_mask,mask)
                running_val_loss.append(loss.item())

        epoch_train_loss = np.mean(running_train_loss)
        print('Train loss: {}'.format(epoch_train_loss))

        epoch_train_cons_loss = np.mean(running_train_cons_loss)
        print('Train consistency loss: {}'.format(epoch_train_cons_loss))

        epoch_val_loss = np.mean(running_val_loss)
        print('Validation loss: {}'.format(epoch_val_loss))

        # saving loss in csv
        loss = [epoch_train_loss, epoch_train_cons_loss, epoch_val_loss] 
        write_to_csv(epoch, loss, csv_filename)

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        with torch.no_grad():
            # Early Stopping by train_cons_loss
            earlystopping(epoch_train_cons_loss)
            if earlystopping.early_stop:
                print("Early stopping")
                torch.save(model.to('cpu').state_dict(), 'weights/'+folname+'_weights/last.pth')
                break
            if earlystopping.counter == 0:
                print(f"Consistency Loss declined to {earlystopping.best_score}")
                # download to CPU
                torch.save(model.to('cpu').state_dict(), 'weights/'+folname+'_weights/best.pth')
                # upload to GPU
                model = model.to(device)

            print(f'Early Stopping Counter = {earlystopping.counter}')