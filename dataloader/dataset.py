import glob, os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# # Unlabeled → Rissbilder
# class make_datapath_list():
#   def __init__(self,folname,first):
#     self.folname = folname
#     self.first = first
#     img_file_path = sorted(glob.glob('data/Train/images/Rissbilder*'))
#     anno_file_path = sorted(glob.glob('data/Train/masks/Rissbilder*'))
    
#     img_file_path2 = sorted(glob.glob('data/Train/images/[!Rissbilder]*'))
#     anno_file_path2 = sorted(glob.glob('data/Train/masks/[!Rissbilder]*'))
    
#     img_file_path3 = sorted(glob.glob('data/Test/images/Rissbilder*'))
#     anno_file_path3 = sorted(glob.glob('data/Test/masks/Rissbilder*'))

#     img_file_path4 = sorted(glob.glob('data/Test/images/[!Rissbilder]*'))
#     anno_file_path4 = sorted(glob.glob('data/Test/masks/[!Rissbilder]*'))

#     if not self.first:
#       mean_file_path = sorted(glob.glob(f'data/unlabeled_mask/{self.folname}/pred_mean_corrected/*'))
#       var_file_path = sorted(glob.glob(f'data/unlabeled_mask/{self.folname}/pred_var/*'))

#     self.train_labeled_file_path = img_file_path2
#     self.train_anno_file_path = anno_file_path2

#     self.train_unlabeled_file_path = img_file_path
#     if not self.first:
#       self.train_unlabeled_mean_path = mean_file_path
#       self.train_unlabeled_var_path = var_file_path

#     self.val_file_path = img_file_path4
#     self.val_anno_file_path = anno_file_path4

#     self.test_file_path = img_file_path3
#     self.test_anno_file_path = anno_file_path3

#   def get_list(self, path_type):
#     if path_type=="train_labeled":
#       file_path = self.train_labeled_file_path
#       anno_path = self.train_anno_file_path
#     elif path_type=="train_unlabeled":
#       file_path = self.train_unlabeled_file_path
#       if not self.first:
#         mean_path = self.train_unlabeled_mean_path
#         var_path = self.train_unlabeled_var_path
#     elif path_type=="val":
#       file_path = self.val_file_path
#       anno_path = self.val_anno_file_path
#     else:
#       file_path = self.test_file_path
#       anno_path = self.test_anno_file_path

#     img_list = []
#     mean_list = []
#     var_list = []
#     anno_list = []
#     if path_type=="train_unlabeled":
      # ##　imgをmean, varに存在する分だけ格納
      # if not self.first:
      #     filenames = [path.lstrip(f'data/unlabeled_mask/{self.folname}/pred_mean_corrected/').rstrip('.jpg') for path in mean_path]
      #     for path in file_path:
      #       if path.lstrip('data/Train/images/').rstrip('.jpg') in filenames:
      #         img_list.append(path)
      #     for path in mean_path:
      #       mean_list.append(path)
      #     for path in var_path:
      #       var_list.append(path)
      #     return img_list, mean_list, var_list
      # else:
      #   for path in file_path:
      #     img_list.append(path)
      #   return img_list
#     else:
#       for path in file_path:
#         img_list.append(path)
#       for path in anno_path:
#         anno_list.append(path)
#       return img_list, anno_list

## Unlabeled → Chundata
class make_datapath_list():
  def __init__(self,folname,first):
    self.folname = folname
    self.first = first
    img_file_path = sorted(glob.glob('data/Train/images/*'))
    anno_file_path = sorted(glob.glob('data/Train/masks/*'))

    img_file_path2 = sorted(glob.glob('data/original_split_resized/*'))
    anno_file_path2 = sorted(glob.glob('data/teacher_split_resized/*'))

    with open("shuffle_indices.txt", "r") as file:
        shuffle_indices = list(map(int, file.read().split()))
    # ランダムな並びを使用してリストを再構築
    img_file_path2 = [img_file_path2[i] for i in shuffle_indices]
    anno_file_path2 = [anno_file_path2[i] for i in shuffle_indices]

    img_file_path3 = sorted(glob.glob('data/Test/images/*'))
    anno_file_path3 = sorted(glob.glob('data/Test/masks/*'))

    if not self.first:
      mean_file_path = sorted(glob.glob(f'data/unlabeled_mask/{self.folname}/pred_mean_corrected/*'))
      var_file_path = sorted(glob.glob(f'data/unlabeled_mask/{self.folname}/pred_var/*'))

    self.train_labeled_file_path = img_file_path
    self.train_anno_file_path = anno_file_path

    self.train_unlabeled_file_path = img_file_path2[:4292]
    if not self.first:
      self.train_unlabeled_mean_path = mean_file_path
      self.train_unlabeled_var_path = var_file_path

    self.val_file_path = img_file_path3
    self.val_anno_file_path = anno_file_path3

    self.test_file_path = img_file_path2[4292:5292]
    self.test_anno_file_path = anno_file_path2[4292:5292]

  def get_list(self, path_type):
    if path_type=="train_labeled":
      file_path = self.train_labeled_file_path
      anno_path = self.train_anno_file_path
    elif path_type=="train_unlabeled":
      file_path = self.train_unlabeled_file_path
      if not self.first:
        mean_path = self.train_unlabeled_mean_path
        var_path = self.train_unlabeled_var_path
    elif path_type=="val":
      file_path = self.val_file_path
      anno_path = self.val_anno_file_path
    else:
      file_path = self.test_file_path
      anno_path = self.test_anno_file_path

    img_list = []
    mean_list = []
    var_list = []
    anno_list = []
    if path_type=="train_unlabeled":
      if not self.first:
        filenames = [mp.replace(f'data/unlabeled_mask/{self.folname}/pred_mean_corrected/', '').rstrip('.jpg') for mp in mean_path]
        for path in file_path:
          # mean path内のファイルと同じ名前のファイルだけリストに格納
          if path.lstrip('data/original_split_resized/').rstrip('.jpg') in filenames:
            img_list.append(path)
        for path in mean_path:
          mean_list.append(path)
        for path in var_path:
          var_list.append(path)
        return img_list, mean_list, var_list
      else:
        for path in file_path:
          img_list.append(path)
        return img_list
    else:
      for path in file_path:
        img_list.append(path)
      for path in anno_path:
        anno_list.append(path)
      return img_list, anno_list


class LabeledDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.label_list[idx]

        # 画像とマスクを読み込む
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        # 前処理を適用
        if self.transform:
            image, label = self.transform(image, label)

        return image, label
      

class LabeledTransform():
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, mask):
        # image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(image)
      
        image = transforms.Resize((self.crop_size, self.crop_size))(image)
        mask = transforms.Resize((self.crop_size, self.crop_size))(mask)

        # w,h = image.size
        # # # ランダムなスケールを1.0~2.0の中で選択する
        # scale = torch.FloatTensor(1).uniform_(1.0, 2.0)

        # new_width = int(round(w * scale.item()))
        # new_height = int(round(h * scale.item()))

        # # # 画像をスケール倍にリサイズする
        # image = transforms.Resize((new_width, new_height))(image)
        # mask = transforms.Resize((new_width, new_height))(mask)

        # # ランダムクロップ
        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        # image = transforms.functional.crop(image, i, j, h, w)
        # mask = transforms.functional.crop(mask, i, j, h, w)

        # 水平反転
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # 垂直反転
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # # # -20~20のランダム回転
        # angle = random.randint(-20, 20)
        # image = transforms.functional.rotate(image,angle)
        # mask = transforms.functional.rotate(mask,angle)

        # テンソルに変換
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        mask = torch.where(mask > 0.5, torch.tensor(1), torch.tensor(0))  # しきい値を超える場合は1、超えない場合は0に変換

        return image, mask
    

class ValLabeledTransform():
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, mask):
        
        image = transforms.Resize((self.crop_size, self.crop_size))(image)
        mask = transforms.Resize((self.crop_size, self.crop_size))(mask)

        # センタークロップ
        # image = transforms.CenterCrop(self.crop_size)(image)
        # mask = transforms.CenterCrop(self.crop_size)(mask)

        # テンソルに変換
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        mask = torch.where(mask > 0.5, torch.tensor(1), torch.tensor(0))

        return image, mask
    

class UnlabeledDataset(Dataset):
    def __init__(self, img_list, mean_list, var_list, transform=None):
        self.img_list = img_list
        self.mean_list = mean_list
        self.var_list = var_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        mean_path = self.mean_list[idx]
        var_path = self.var_list[idx]


        # 画像を読み込む
        image = Image.open(img_path).convert('RGB')
        mean = Image.open(mean_path).convert('L')
        var = Image.fromarray(torch.load(var_path).numpy())

        # 前処理を適用
        if self.transform:
            image, mean, var = self.transform(image, mean, var)
        return image, mean, var
    
    
class UnlabeledTransform():
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, mean, var):
        # image = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)(image)
      
        image = transforms.Resize((self.crop_size, self.crop_size))(image)
        mean = transforms.Resize((self.crop_size, self.crop_size))(mean)
        var = transforms.Resize((self.crop_size, self.crop_size))(var)

        # if self.scaling:
        #     # # ランダムなスケールを1.0~2.0の中で選択する
        #     scale = torch.FloatTensor(1).uniform_(1.0, 2.0)

        #     w, h = image.size
        #     new_width = int(round(w * scale.item()))
        #     new_height = int(round(h * scale.item()))

        #     # # 画像をスケール倍にリサイズする
        #     image = transforms.Resize((new_width, new_height))(image)
        #     mean = transforms.Resize((new_width, new_height))(mean)
        #     var = transforms.Resize((new_width, new_height))(var)

        # # ランダムクロップ
        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        # image = transforms.functional.crop(image, i, j, h, w)
        # mean = transforms.functional.crop(mean, i, j, h, w)
        # var = transforms.functional.crop(var, i, j, h, w)

        # 水平反転
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mean = transforms.functional.hflip(mean)
            var = transforms.functional.hflip(var)

        # 垂直反転
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mean = transforms.functional.vflip(mean)
            var = transforms.functional.vflip(var)
            
        # angle = random.randint(-20, 20)
        # image = transforms.functional.rotate(image,angle)
        # mean = transforms.functional.rotate(mean,angle)
        # var = transforms.functional.rotate(var,angle)

        # テンソルに変換
        image = transforms.ToTensor()(image)
        mean = transforms.ToTensor()(mean)
        mean = torch.where(mean > 0.5, torch.tensor(1), torch.tensor(0))
        var = transforms.ToTensor()(var)

        return image, mean, var


class UnlabeledDataset2(Dataset):
    def __init__(self, img_list, transform=None):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        # 画像を読み込む
        image = Image.open(img_path).convert('RGB')

        # 前処理を適用
        if self.transform:
            image = self.transform(image)
        return image

class UnlabeledTransform2():
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        image = transforms.Resize((self.crop_size, self.crop_size))(image)
        # if self.scaling:
        #     # # ランダムなスケールを1.0~2.0の中で選択する
        #     scale = torch.FloatTensor(1).uniform_(1.0, 2.0)

        #     w, h = image.size
        #     new_width = int(round(w * scale.item()))
        #     new_height = int(round(h * scale.item()))

        #     # # 画像をスケール倍にリサイズする
        #     image = transforms.Resize((new_width, new_height))(image)

        # # ランダムクロップ
        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        # image = transforms.functional.crop(image, i, j, h, w)

        # if self.flip:
        #     # 水平反転
        #     if random.random() > 0.5:
        #         image = transforms.functional.hflip(image)

        #     # 垂直反転
        #     if random.random() > 0.5:
        #         image = transforms.functional.vflip(image)

        # テンソルに変換
        image = transforms.ToTensor()(image)

        return image
    

# # define datapath for fully supervised learning
# class make_datapath_list_supervised():
#   def __init__(self):
#     img_file_path = sorted(glob.glob('data/Train/images/Rissbilder*'))
#     anno_file_path = sorted(glob.glob('data/Train/masks/Rissbilder*'))

#     indices = np.arange(len(img_file_path))
#     np.random.shuffle(indices)
#     img_file_path = [img_file_path[i] for i in indices]
#     anno_file_path = [anno_file_path[i] for i in indices]
    
#     img_file_path2 = sorted(glob.glob('data/Test/images/Rissbilder*'))
#     anno_file_path2 = sorted(glob.glob('data/Test/masks/Rissbilder*'))

#     self.train_labeled_file_path = img_file_path[:3000]
#     self.train_anno_file_path = anno_file_path[:3000]

#     self.val_file_path = img_file_path[3000:3440]
#     self.val_anno_file_path = anno_file_path[3000:3440]

#     self.test_file_path = img_file_path2
#     self.test_anno_file_path = anno_file_path2

#   def get_list(self, path_type):
#     if path_type=="train_labeled":
#       file_path = self.train_labeled_file_path
#       anno_path = self.train_anno_file_path
#     elif path_type=="val":
#       file_path = self.val_file_path
#       anno_path = self.val_anno_file_path
#     else:
#       file_path = self.test_file_path
#       anno_path = self.test_anno_file_path

#     img_list = []
#     anno_list = []
#     for path in file_path:
#       img_list.append(path)
#     for path in anno_path:
#       anno_list.append(path)
#     return img_list, anno_list

class make_datapath_list_supervised():
  def __init__(self):
    img_file_path = sorted(glob.glob('data/Train/images/*'))
    anno_file_path = sorted(glob.glob('data/Train/masks/*'))

    img_file_path2 = sorted(glob.glob('data/original_split_resized/*'))
    anno_file_path2 = sorted(glob.glob('data/teacher_split_resized/*'))

    with open("shuffle_indices.txt", "r") as file:
      shuffle_indices = list(map(int, file.read().split()))
    # ランダムな並びを使用してリストを再構築
    img_file_path2 = [img_file_path2[i] for i in shuffle_indices]
    anno_file_path2 = [anno_file_path2[i] for i in shuffle_indices]

    self.train_labeled_file_path = img_file_path[:] + img_file_path2[:3500]
    self.train_anno_file_path = anno_file_path[:] + anno_file_path2[:3500]

    self.val_file_path = img_file_path2[3500:4292]
    self.val_anno_file_path = anno_file_path2[3500:4292]

    self.test_file_path = img_file_path2[4292:5292]
    self.test_anno_file_path = anno_file_path2[4292:5292]

  def get_list(self, path_type):
    if path_type=="train_labeled":
      file_path = self.train_labeled_file_path
      anno_path = self.train_anno_file_path
    elif path_type=="val":
      file_path = self.val_file_path
      anno_path = self.val_anno_file_path
    else:
      file_path = self.test_file_path
      anno_path = self.test_anno_file_path

    img_list = []
    anno_list = []
    for path in file_path:
      img_list.append(path)
    for path in anno_path:
      anno_list.append(path)
    return img_list, anno_list