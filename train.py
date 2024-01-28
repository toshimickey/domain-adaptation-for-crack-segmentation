import os
import time
import numpy as np
import csv
import torch
import torch.utils.data as data
from tqdm import tqdm
from utils.loss_function import DiceBCELoss, BayesBCELoss, RMSELoss
from models.bayesian_deeplab import DeepLabv3plusModel
from models.bayesian_unet import Unet256
from utils.module import EarlyStopping, write_to_csv
from dataloader.dataset import make_datapath_list, make_datapath_list_supervised, LabeledDataset, LabeledTransform, ValLabeledTransform, UnlabeledDataset, UnlabeledTransform


def train(former_folname, folname, first=False, net="deeplab", batch_size=64, num_workers=2, epochs=300, alpha=100, beta=10, crop_size=256, supervised=False, cons_reg=False):
    # make dataloader
    if not supervised:
        makepath = make_datapath_list(former_folname, first)
    else:
        makepath = make_datapath_list_supervised()
    train_labeled_img_list, train_labeled_anno_list = makepath.get_list("train_labeled")
    if not first:
        train_unlabeled_img_list, train_unlabeled_mean_list, train_unlabeled_var_list = makepath.get_list("train_unlabeled")
    val_img_list, val_anno_list = makepath.get_list("val")

    train_labeled_dataset = LabeledDataset(train_labeled_img_list, train_labeled_anno_list, transform=LabeledTransform(crop_size=crop_size))
    val_dataset = LabeledDataset(val_img_list, val_anno_list, transform=ValLabeledTransform(crop_size=crop_size))
    if not first:
        train_unlabeled_dataset = UnlabeledDataset(train_unlabeled_img_list, train_unlabeled_mean_list, train_unlabeled_var_list, transform=UnlabeledTransform(crop_size=crop_size))

    train_labeled_dataloader = data.DataLoader(
        train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    if not first:
        train_unlabeled_dataloader = data.DataLoader(
            train_unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_dataloader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # define loss function
    criterion1 = DiceBCELoss()
    criterion2 = BayesBCELoss(alpha,beta)
    cons_criterion = RMSELoss()

    # define model
    if net == "deeplab":
        model_wrapper = DeepLabv3plusModel()
        model = model_wrapper.get_model()
    else:
        model = Unet256((3, crop_size, crop_size))

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    start_epoch = 0
    earlystopping = EarlyStopping(patience=50)

    parts = folname.split("_")
    project, iter = parts[0],parts[1]

    os.makedirs('weights/'+ project + '/'+ iter +'_weights', exist_ok=True)
    os.makedirs('loss/'+ project + '/'+ iter +'_loss', exist_ok=True)

    csv_filename = 'loss/'+ project + '/'+ iter +'_loss/loss.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        if first:
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])
        elif cons_reg:
            writer.writerow(['Epoch', 'Train Loss', 'Train Unlabel Loss', 'Train Consistency Loss', 'Validation Loss'])
        else:
            writer.writerow(['Epoch', 'Train Loss', 'Train Unlabel Loss', 'Validation Loss'])
        
    model = model.to(device)

    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        start_time = time.time()

        running_train_loss = []
        running_train_unlabel_loss = []
        running_train_cons_loss = []
        model.train()
        for image,mask in tqdm(train_labeled_dataloader):
            image = image.to(device,dtype=torch.float)
            mask = mask.to(device,dtype=torch.float)
            pred_mask = model.forward(image)
            loss = criterion1(pred_mask,mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())
        
        if not first:
            for image, mean, var in tqdm(train_unlabeled_dataloader):
                image = image.to(device,dtype=torch.float)
                mean = mean.to(device,dtype=torch.float)
                var = var.to(device,dtype=torch.float)

                pred_mask = model.forward(image)
                loss = criterion2(pred_mask,mean,var) #var
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_train_unlabel_loss.append(loss.item())
                
                if cons_reg:
                    pred_mask1 = model.forward(image)
                    pred_mask2 = model.forward(image)
                    loss = cons_criterion(pred_mask1, pred_mask2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_train_cons_loss.append(loss.item())

        running_val_loss = []
        model.eval()
        with torch.no_grad():
            for image,mask in tqdm(val_dataloader):
                image = image.to(device,dtype=torch.float)
                mask = mask.to(device,dtype=torch.float)
                pred_mask = model.forward(image)
                loss = criterion1(pred_mask,mask)
                running_val_loss.append(loss.item())

        epoch_train_loss = np.mean(running_train_loss)
        #print('Train loss: {}'.format(epoch_train_loss))

        if not first:
            epoch_train_unlabel_loss = np.mean(running_train_unlabel_loss)
            #print('Train consistency loss: {}'.format(epoch_train_cons_loss))
            
        if cons_reg:
            epoch_train_cons_loss = np.mean(running_train_cons_loss)

        epoch_val_loss = np.mean(running_val_loss)
        #print('Validation loss: {}'.format(epoch_val_loss))

        # saving loss in csv
        if first:
            loss = [epoch_train_loss, epoch_val_loss] 
            write_to_csv(epoch+1, loss, csv_filename)
        elif cons_reg:
            loss = [epoch_train_loss, epoch_train_unlabel_loss, epoch_train_cons_loss, epoch_val_loss] 
            write_to_csv(epoch+1, loss, csv_filename)
        else:
            loss = [epoch_train_loss, epoch_train_unlabel_loss, epoch_val_loss] 
            write_to_csv(epoch+1, loss, csv_filename)

        time_elapsed = time.time() - start_time
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        with torch.no_grad():
            # Early Stopping
            if first:
                earlystopping(epoch_val_loss)
            else:
                earlystopping(epoch_train_unlabel_loss)

            if earlystopping.early_stop:
                print("Early stopping")
                break
            if earlystopping.counter == 0:
                if not first:
                    print(f"Consistency Loss declined to {earlystopping.best_score}")
                else:
                    print(f"Validation Loss declined to {earlystopping.best_score}")
                # download to CPU
                torch.save(model.to('cpu').state_dict(), 'weights/'+ project + '/'+ iter +'_weights/best.pth')
                # upload to GPU
                model = model.to(device)

            print(f'Early Stopping Counter = {earlystopping.counter}')

    torch.save(model.to('cpu').state_dict(), 'weights/'+ project + '/'+ iter +'_weights/last.pth')