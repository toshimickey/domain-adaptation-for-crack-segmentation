import pickle
from dataloader.dataset import make_datapath_list, LabeledDataset, LabeledTransform, ValLabeledTransform, UnlabeledDataset, UnlabeledTransform
import torch.utils.data as data
import torch
from utils.loss_function import DiceBCELoss, BayesBCELoss
import segmentation_models_pytorch as smp
from models.bayesian_deeplab import Dropout2d, DeepLabv3plusModel
from utils.earlystopping import EarlyStopping
import os
import time
import numpy as np


makepath = make_datapath_list()
train_labeled_img_list, train_labeled_anno_list = makepath.get_list("train_labeled")
train_unlabeled_img_list, train_unlabeled_mean_list, train_unlabeled_var_list = makepath.get_list("train_unlabeled")
val_img_list, val_anno_list = makepath.get_list("val")

train_labeled_dataset = LabeledDataset(train_labeled_img_list, train_labeled_anno_list, transform=LabeledTransform(crop_size=256))
val_dataset = LabeledDataset(val_img_list, val_anno_list, transform=ValLabeledTransform(crop_size=256))
train_unlabeled_dataset = UnlabeledDataset(train_unlabeled_img_list, train_unlabeled_mean_list, train_unlabeled_var_list, transform=UnlabeledTransform(crop_size=256, rotation=True))

train_labeled_dataloader = data.DataLoader(
    train_labeled_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
train_unlabeled_dataloader = data.DataLoader(
    train_unlabeled_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = data.DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

class_criterion = DiceBCELoss()
cons_criterion = BayesBCELoss(alpha=1000)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_wrapper = DeepLabv3plusModel(device)
model = model_wrapper.get_model()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
#model.parameters()は訓練対象のパラメータ

epochs = 1000
train_loss = []
train_cons_loss = []
val_loss = []
date = '231010'
start_epoch = 0
earlystopping = EarlyStopping(patience=50)

# continueする場合、ここを埋める。それ以外だとコメントアウト
model_path = 'weights/231010_weights/model_46.pth'
model.load_state_dict(torch.load(model_path))
start_epoch = 46
earlystopping.counter = 0
earlystopping.best_score =  0.014148265207649406


os.makedirs('weights/'+ date +'_weights', exist_ok=True)
os.makedirs('loss/'+ date +'_loss', exist_ok=True)
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
        loss = cons_criterion(pred_mask,mean,var)*10
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
    train_loss.append(epoch_train_loss)

    epoch_train_cons_loss = np.mean(running_train_cons_loss)
    print('Train consistency loss: {}'.format(epoch_train_cons_loss))
    train_cons_loss.append(epoch_train_cons_loss)

    epoch_val_loss = np.mean(running_val_loss)
    print('Validation loss: {}'.format(epoch_val_loss))
    val_loss.append(epoch_val_loss)

    # saving loss
    loss = [epoch_train_loss, epoch_train_cons_loss, epoch_val_loss]
    with open('loss/'+date+f'_loss/{epoch+1}.pickle', mode='wb') as fo:
        pickle.dump(loss, fo)

    time_elapsed = time.time() - start_time
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    with torch.no_grad():
        # train_cons_lossでEarly Stopping
        earlystopping(epoch_train_cons_loss)
        if earlystopping.early_stop:
            print("Early stopping")
            break
        if earlystopping.counter == 0:
            print(f"Consistency Loss declined to {earlystopping.best_score}")
            # download to CPU
            torch.save(model.to('cpu').state_dict(), 'weights/'+date+'_weights/model_'+str(epoch+1)+'.pth')
            # upload to GPU
            model = model.to(device)

        print(f'Early Stopping Counter = {earlystopping.counter}')