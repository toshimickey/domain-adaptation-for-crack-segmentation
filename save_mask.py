import torch
import torch.utils.data as data
from dataloader.dataset import make_datapath_list, UnlabeledDataset2, UnlabeledTransform2
from models.bayesian_deeplab import DeepLabv3plusModel
from models.bayesian_unet import Unet256
import os
import numpy as np
from PIL import Image
from scipy.ndimage import label
from skimage import measure
import torchvision.transforms as transforms

def process_image(image, area_threshold=100, compactness_threshold=0.015, eccentricity_threshold=0.95):
    # 画像を2値化する→この処理はsigmoid有無に関わらずそのままでOK
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

def save_mask(former_folname, folname, net="deeplab", batch_size=64, num_workers=2):
    # unlabeled dataに対するpred_mean, pred_varを保存
    makepath = make_datapath_list(former_folname, first=True)
    train_unlabeled_img_list = makepath.get_list("train_unlabeled")
    train_unlabeled_dataset = UnlabeledDataset2(train_unlabeled_img_list, transform=UnlabeledTransform2(crop_size=512, flip=False, scaling=False))
    train_unlabeled_dataloader = data.DataLoader(
        train_unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if net == "deeplab":
        model_wrapper = DeepLabv3plusModel()
        model = model_wrapper.get_model()
    else:
        model = Unet256((3, 512, 512))

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    img_filename = sorted(os.listdir('data/original_split_resized'))

    os.makedirs(f'data/unlabeled_mask/{folname}/pred_mean')
    os.makedirs(f'data/unlabeled_mask/{folname}/pred_mean_corrected')
    os.makedirs(f'data/unlabeled_mask/{folname}/pred_var')

    n_samples = 100
    count = 0
    flag = False
    model.to(device)
    model_path = f'weights/{folname}_weights/best.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for image in train_unlabeled_dataloader:
            image = image.to(device,dtype=torch.float)

            # sampling for n times
            preds = []
            for _ in range(n_samples):
                pred = model.forward(image).unsqueeze(0)
                preds.append(pred)
            preds = torch.cat(preds)
            pred_mean = torch.mean(preds, dim=0)
            # pred_varはpredsをsigmoidに通した後に分散をとる
            pred_var = torch.var(torch.sigmoid(preds), dim=0)

            for j in range(batch_size):
                # pred_mean[j]をsigmoidに通してjpgとして保存
                image_mean = torch.sigmoid(pred_mean[j]).cpu().detach().numpy()*255
                # image_mean = pred_mean[j].cpu().detach().clamp(0, 1).numpy()*255
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

    former_path = f'data/unlabeled_mask/{folname}/pred_mean/'
    latter_path = f'data/unlabeled_mask/{folname}/pred_mean_corrected/'
    files = sorted(os.listdir(former_path))

    for i in range(len(files)):
        image = Image.open(former_path+files[i])
        image = np.array(image)
        image_corrected = process_image(image)
        image_tosave = Image.fromarray(image_corrected.astype('uint8'))
        image_tosave.save(latter_path+files[i])