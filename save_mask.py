import torch
import torch.utils.data as data
from dataloader.dataset import make_datapath_list, make_datapath_list_fromJson, UnlabeledDataset, UnlabeledTransform
from models.bayesian_deeplab import DeepLabv3plusModel
from models.bayesian_unet import Unet256
import os, glob
import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage import measure
import torchvision.transforms as transforms
from tqdm import tqdm
import pandas as pd
import json

def itemgetter(list,idx):
    output = []
    for i in idx:
        output.append(list[i])
    return output

def mean_anomaly_score(feature, df):
    scores = []
    for dataset in df.dataset.unique():
        df_subset = df[df['dataset']==dataset]
        data = df_subset['area_by_perimeter'].values
        
        # 平均と標準偏差の計算
        mean = np.mean(data)
        std = np.std(data)
        
        score = ((feature - mean) / std)**2
        scores.append(score)
    return sum(scores)/len(scores)

# input(image, df)→ output(image_corrected)
def process_image(image, df):
    
    binary_image = image > 0
    labeled_image, num_labels = label(binary_image)
    props = measure.regionprops(labeled_image)

    # 条件を満たさない連結要素を特定し、黒く塗りつぶす
    for prop in props:
        if prop.area < 100 or prop.eccentricity < 0.5 or prop.perimeter < 10 \
                or prop.axis_minor_length <= 0 or prop.axis_major_length <= 0: 
            labeled_image[labeled_image == prop.label] = 0
        else:
            feature = prop.area/(prop.perimeter**2)
            if mean_anomaly_score(feature, df) > 7.88:
                labeled_image[labeled_image == prop.label] = 0
            
    # 処理後の画像を返す
    crack_label = labeled_image > 0
    processed_image = np.where(crack_label, image, 0)
    return processed_image

# unlabeled dataに対するpred_mean, pred_varを保存
def save_mask(former_folname, folname, JsonDataSplit=False, target_dataset='chun', useStableDiffusion=False, net="deeplab", batch_size=64, num_workers=2, crop_size=256, pred_max=False, save_num=None):
    first = True
    if not JsonDataSplit:
        makepath = make_datapath_list(former_folname, first, target_dataset, useStableDiffusion)
    else:
        makepath = make_datapath_list_fromJson(former_folname, first)
    
    train_unlabeled_img_list = makepath.get_list("train_unlabeled")
    train_unlabeled_dataset = UnlabeledDataset(train_unlabeled_img_list, transform=UnlabeledTransform(crop_size=crop_size))
    train_unlabeled_dataloader = data.DataLoader(
        train_unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if net == "deeplab":
        model_wrapper = DeepLabv3plusModel()
        model = model_wrapper.get_model()
    else:
        model = Unet256((3, crop_size, crop_size))

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if JsonDataSplit:
        img_file_path = sorted(glob.glob('data/Train/images/*')) + sorted(glob.glob('data/original_split_resized/*'))
        with open('data_split.json', 'r') as json_file:
            idx = json.load(json_file)
        img_file_path3 = sorted(itemgetter(img_file_path,idx[2]))
        img_filename = [os.path.basename(file) for file in img_file_path3]
    elif useStableDiffusion:
        # stable diffusionのpath名
        img_file_path = sorted(glob.glob('data/2023-12-25/*') + glob.glob('data/sampled100/*'))
        img_filename = [os.path.basename(file) for file in img_file_path]
    else:
        if target_dataset == 'chun':
            img_filename = sorted(os.listdir('data/original_split_resized'))
            img_filename = sorted(img_filename, key=lambda x: (int(x.split('_')[0].lstrip('c')), int(x.split('_')[1])))
        elif target_dataset == 'volker':
            img_file_path = sorted(glob.glob('data/Train/images/Volker*'))
            img_filename = [file.lstrip('data/Train/images/') for file in img_file_path]
    
    confidence_list = []

    os.makedirs(f'data/unlabeled_mask/{folname}/pred_mean')
    os.makedirs(f'data/unlabeled_mask/{folname}/pred_mean_corrected')
    os.makedirs(f'data/unlabeled_mask/{folname}/pred_var')

    n_samples = 100
    count = 0
    flag = False
    model.to(device)
    parts = folname.split("_")
    project, iter = parts[0],parts[1]

    model_path = 'weights/'+ project + '/'+ iter +'_weights/best.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for image in tqdm(train_unlabeled_dataloader):
            image = image.to(device,dtype=torch.float)

            # sampling for n times
            preds = []
            for _ in range(n_samples):
                pred = model.forward(image).unsqueeze(0)
                preds.append(pred)
            preds = torch.cat(preds)
            
            # Recall高めるため、サンプルの最大値をpred_meanとする
            if pred_max:
                pred_mean = torch.max(preds, dim=0).values
            else:
                pred_mean = torch.mean(preds, dim=0)
            
            # confidence listに確信度を格納 pred_mean (64,1,256,256)
            pred_conf = torch.max(pred_mean, 1-pred_mean).view(pred_mean.size(0), -1)
            confidence = torch.mean(pred_conf, dim=1).tolist()
            confidence_list = confidence_list + confidence

            # pred_varはpredsをsigmoidに通した後に分散をとる
            pred_var = torch.var(torch.sigmoid(preds), dim=0)

            for j in range(batch_size):
                # pred_mean[j]をsigmoidに通してjpgとして保存
                image_mean = torch.sigmoid(pred_mean[j]).cpu().detach().numpy()*255
                image_mean = Image.fromarray(image_mean[0].astype('uint8'))
                image_mean.save(f'data/unlabeled_mask/{folname}/pred_mean/{img_filename[count]}')

                # pred_var[j]をtensor.ptとして保存
                image_var = pred_var[j][0].cpu()
                torch.save(image_var, f'data/unlabeled_mask/{folname}/pred_var/{img_filename[count]}'.rstrip('jpg')+'pt')

                count += 1
                if count == len(train_unlabeled_dataset):
                    flag = True
                    break
            if flag:
                break
    
    # confidenceの大きいデータsave_num個以外を削除
    if save_num is not None:
        indexed_confidence = list(enumerate(confidence_list))
        sorted_confidence = sorted(indexed_confidence, key=lambda x: x[1], reverse=True)
        # indicesはconfidenceの小さいデータ
        indices = [index for index, _ in sorted_confidence[save_num:]]
        for index in indices:
            os.remove(f'data/unlabeled_mask/{folname}/pred_mean/{img_filename[index]}')
            os.remove(f'data/unlabeled_mask/{folname}/pred_var/{img_filename[index]}'.rstrip('jpg')+'pt')
    
    
    former_path = f'data/unlabeled_mask/{folname}/pred_mean/'
    latter_path = f'data/unlabeled_mask/{folname}/pred_mean_corrected/'
    files = sorted(os.listdir(former_path))
    
    train_feat_df = pd.read_csv("train_feat_df.csv")

    for i in tqdm(range(len(files))):
        image = Image.open(former_path+files[i])
        image = np.array(image)
        image_corrected = process_image(image, train_feat_df)
        image_tosave = Image.fromarray(image_corrected.astype('uint8'))
        image_tosave.save(latter_path+files[i])