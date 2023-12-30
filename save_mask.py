import torch
import torch.utils.data as data
from dataloader.dataset import make_datapath_list, UnlabeledDataset2, UnlabeledTransform2
from models.bayesian_deeplab import DeepLabv3plusModel
from models.bayesian_unet import Unet256
import os, glob
import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage import measure
import torchvision.transforms as transforms
from tqdm import tqdm

def process_image(image, area_threshold=100, compactness_threshold=0.015, eccentricity_threshold=0.95):
    # 画像を2値化する
    binary_image = image > 0

    # 連結要素のラベリングを行う
    labeled_image, num_labels = label(binary_image)

    # 各連結要素の特徴量を計算する
    properties = measure.regionprops(labeled_image)

    # 条件を満たさない連結要素を特定し、黒く塗りつぶす
    for prop in properties:
        area = prop.area
        perimeter = prop.perimeter
        eccentricity = prop.eccentricity

        if area < area_threshold or area / (perimeter ** 2) > compactness_threshold: #or eccentricity < eccentricity_threshold:
            labeled_image[labeled_image == prop.label] = 0

    # 処理後の画像を返す
    crack_label = labeled_image > 0
    processed_image = np.where(crack_label, image, 0)
    return processed_image


def save_mask(former_folname, folname, net="deeplab", batch_size=64, num_workers=2, crop_size=256, pred_max=False, save_num=None):
    # unlabeled dataに対するpred_mean, pred_varを保存
    makepath = make_datapath_list(former_folname, first=True)
    train_unlabeled_img_list = makepath.get_list("train_unlabeled")
    train_unlabeled_dataset = UnlabeledDataset2(train_unlabeled_img_list, transform=UnlabeledTransform2(crop_size=crop_size))
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

    img_filename = sorted(os.listdir('data/original_split_resized'))
    img_filename = sorted(img_filename, key=lambda x: (int(x.split('_')[0].lstrip('c')), int(x.split('_')[1])))
    
    # img_file_path = sorted(glob.glob('data/Train/images/Volker*'))
    # img_filename = [file.lstrip('data/Train/images/') for file in img_file_path]
    
    # img_file_path = sorted(glob.glob('data/2023-12-27/*'))
    # img_filename = [os.path.basename(file) for file in img_file_path]
    
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

    for i in tqdm(range(len(files))):
        image = Image.open(former_path+files[i])
        image = np.array(image)
        image_corrected = process_image(image)
        image_tosave = Image.fromarray(image_corrected.astype('uint8'))
        image_tosave.save(latter_path+files[i])