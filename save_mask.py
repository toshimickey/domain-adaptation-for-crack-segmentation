#　実行前にunlabeled loaderの定義を見直す
# unlabeled dataに対するpred_mean, pred_varを保存
# 同じファイル名で保存
#img_file_path = sorted(glob.glob('data/Chundata/original_split/*'))
img_filename = sorted(os.listdir('data/Chundata/original_split'))
train_unlabeled_dataloader = data.DataLoader(
    train_unlabeled_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

n_samples = 100
count = 0
flag = False
model.to(device)
model.eval()
with torch.no_grad():
    for image, mean, var in train_unlabeled_dataloader:
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

        for j in range(16):
            # pred_mean[j]をsigmoidに通してjpgとして保存
            image_mean = torch.sigmoid(pred_mean[j]).cpu().detach().numpy()*255
            # image_mean = pred_mean[j].cpu().detach().clamp(0, 1).numpy()*255
            image_mean = Image.fromarray(image_mean[0].astype('uint8'))
            image_mean.save(f'data/Chundata_unlabeled_mask/{fol_name}/pred_mean/{img_filename[count]}')
            # pred_var[j]をtensor.ptとして保存
            image_var = pred_var[j][0].cpu()
            torch.save(image_var, f'data/Chundata_unlabeled_mask/{fol_name}/pred_var/{img_filename[count]}'.rstrip('jpg')+'pt')

            count += 1
            if count == 4292:
              flag = True
              break
        if flag:
            break

from scipy.ndimage import label
from skimage import measure
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

former_path = f'data/Chundata_unlabeled_mask/{fol_name}/pred_mean/'
latter_path = f'data/Chundata_unlabeled_mask/{fol_name}/pred_mean_corrected/'
files = sorted(os.listdir(former_path))

for i in range(len(files)):
    image = Image.open(former_path+files[i])
    image = np.array(image)
    image_corrected = process_image(image)
    image_tosave = Image.fromarray(image_corrected.astype('uint8'))
    image_tosave.save(latter_path+files[i])