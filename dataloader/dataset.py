import glob, os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json

def itemgetter(list,idx):
    output = []
    for i in idx:
        output.append(list[i])
    return output

# data_split.jsonでデータ構成を定義したデータセット
class make_datapath_list_fromJson():
  def __init__(self,folname,first):
    self.folname = folname
    self.first = first
    img_file_path = sorted(glob.glob('data/Train/images/*')) + sorted(glob.glob('data/original_split_resized/*'))
    anno_file_path = sorted(glob.glob('data/Train/masks/*')) + sorted(glob.glob('data/teacher_split_resized/*'))
    
    with open('data_split.json', 'r') as json_file:
      idx = json.load(json_file)

    img_file_path1 = itemgetter(img_file_path,idx[0])
    anno_file_path1 = itemgetter(anno_file_path,idx[0])
    img_file_path2 = itemgetter(img_file_path,idx[1])
    anno_file_path2 = itemgetter(anno_file_path,idx[1])
    img_file_path3 = sorted(itemgetter(img_file_path,idx[2]))
    anno_file_path3 = itemgetter(anno_file_path,idx[2])
    img_file_path4 = itemgetter(img_file_path,idx[3])
    anno_file_path4 = itemgetter(anno_file_path,idx[3])
    
    if not self.first:
      mean_file_path = sorted(glob.glob(f'data/unlabeled_mask/{self.folname}/pred_mean_corrected/*'))
      var_file_path = sorted(glob.glob(f'data/unlabeled_mask/{self.folname}/pred_var/*'))

    self.train_labeled_file_path = img_file_path1
    self.train_anno_file_path = anno_file_path1

    self.train_unlabeled_file_path = img_file_path3
    if not self.first:
      self.train_unlabeled_mean_path = mean_file_path
      self.train_unlabeled_var_path = var_file_path

    self.val_file_path = img_file_path2
    self.val_anno_file_path = anno_file_path2

    self.test_file_path = img_file_path4
    self.test_anno_file_path = anno_file_path4

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
        for path in file_path:
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


# ドメイン適応のためのデータセット
class make_datapath_list():
  def __init__(self, folname, first, target_dataset, useStableDiffusion=False):
    self.first = first
    if target_dataset == 'chun':
      img_file_path = sorted(glob.glob('data/Train/images/*'))
      anno_file_path = sorted(glob.glob('data/Train/masks/*'))

      img_file_path2 = sorted(glob.glob('data/original_split_resized/*'))
      anno_file_path2 = sorted(glob.glob('data/teacher_split_resized/*'))      
      img_file_path2 = sorted(img_file_path2, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))
      anno_file_path2 = sorted(anno_file_path2, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))

      img_file_path3 = sorted(glob.glob('data/Test/images/*'))
      anno_file_path3 = sorted(glob.glob('data/Test/masks/*'))
      
      # Stable Diffusion生成画像の格納先
      img_file_path4 = sorted(glob.glob('data/2023-12-25/*') + glob.glob('data/sampled100/*'))

      if not first:
        mean_file_path = sorted(glob.glob(f'data/unlabeled_mask/{folname}/pred_mean_corrected/*'))
        var_file_path = sorted(glob.glob(f'data/unlabeled_mask/{folname}/pred_var/*'))
        if not useStableDiffusion:
          mean_file_path = sorted(mean_file_path, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))
          var_file_path = sorted(var_file_path, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))          

      self.train_labeled_file_path = img_file_path
      self.train_anno_file_path = anno_file_path
      
      if not useStableDiffusion:
        self.train_unlabeled_file_path = img_file_path2[:4212]
      else:
        self.train_unlabeled_file_path = img_file_path4
  
      if not first:
        self.train_unlabeled_mean_path = mean_file_path
        self.train_unlabeled_var_path = var_file_path

      self.val_file_path = img_file_path3
      self.val_anno_file_path = anno_file_path3

      self.test_file_path = img_file_path2[4212:5292]
      self.test_anno_file_path = anno_file_path2[4212:5292]
      
    elif target_dataset == 'volker':
      img_file_path = sorted(glob.glob('data/Train/images/Volker*'))
      
      img_file_path2 = sorted(glob.glob('data/Train/images/[!Volker]*'))
      anno_file_path2 = sorted(glob.glob('data/Train/masks/[!Volker]*'))
      
      img_file_path3 = sorted(glob.glob('data/Test/images/Volker*'))
      anno_file_path3 = sorted(glob.glob('data/Test/masks/Volker*'))

      img_file_path4 = sorted(glob.glob('data/Test/images/[!Volker]*'))
      anno_file_path4 = sorted(glob.glob('data/Test/masks/[!Volker]*'))
      
      img_file_path5 = sorted(glob.glob('data/original_split_resized/*'))
      anno_file_path5 = sorted(glob.glob('data/teacher_split_resized/*'))
      img_file_path5 = sorted(img_file_path5, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))
      anno_file_path5 = sorted(anno_file_path5, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))

      if not first:
        mean_file_path = sorted(glob.glob(f'data/unlabeled_mask/{folname}/pred_mean_corrected/*'))
        var_file_path = sorted(glob.glob(f'data/unlabeled_mask/{folname}/pred_var/*'))

      self.train_labeled_file_path = img_file_path2+img_file_path5[:4212]
      self.train_anno_file_path = anno_file_path2+anno_file_path5[:4212]

      self.train_unlabeled_file_path = img_file_path
      if not first:
        self.train_unlabeled_mean_path = mean_file_path
        self.train_unlabeled_var_path = var_file_path

      self.val_file_path = img_file_path4+img_file_path5[4212:5292]
      self.val_anno_file_path = anno_file_path4+anno_file_path5[4212:5292]

      self.test_file_path = img_file_path3
      self.test_anno_file_path = anno_file_path3

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
          for path in file_path:
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

#　教師あり学習のためのデータセット
class make_datapath_list_supervised():
  def __init__(self, target_dataset):
    if target_dataset == 'chun':
      img_file_path = sorted(glob.glob('data/Train/images/*'))
      anno_file_path = sorted(glob.glob('data/Train/masks/*'))

      img_file_path2 = sorted(glob.glob('data/original_split_resized/*'))
      anno_file_path2 = sorted(glob.glob('data/teacher_split_resized/*'))      
      img_file_path2 = sorted(img_file_path2, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))
      anno_file_path2 = sorted(anno_file_path2, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))

      img_file_path3 = sorted(glob.glob('data/Test/images/*'))
      anno_file_path3 = sorted(glob.glob('data/Test/masks/*'))
      
      self.train_labeled_file_path = img_file_path[:] + img_file_path2[:3672] + img_file_path3[:]
      self.train_anno_file_path = anno_file_path[:] + anno_file_path2[:3672] + anno_file_path3[:]

      self.val_file_path = img_file_path2[3672:4212]
      self.val_anno_file_path = anno_file_path2[3672:4212]

      self.test_file_path = img_file_path2[4212:5292]
      self.test_anno_file_path = anno_file_path2[4212:5292]
      
    elif target_dataset == 'volker':
      
      img_file_path = sorted(glob.glob('data/Train/images/Volker*'))
      anno_file_path = sorted(glob.glob('data/Train/masks/Volker*'))

      img_file_path2 = sorted(glob.glob('data/original_split_resized/*'))
      anno_file_path2 = sorted(glob.glob('data/teacher_split_resized/*'))
      img_file_path2 = sorted(img_file_path2, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))
      anno_file_path2 = sorted(anno_file_path2, key=lambda x: (int(os.path.basename(x).split('_')[0].lstrip('c')), int(os.path.basename(x).split('_')[1])))
      
      img_file_path3 = sorted(glob.glob('data/Train/images/[!Volker]*'))
      anno_file_path3 = sorted(glob.glob('data/Train/masks/[!Volker]*'))
      
      img_file_path4 = sorted(glob.glob('data/Test/images/Volker*'))
      anno_file_path4 = sorted(glob.glob('data/Test/masks/Volker*'))
      
      self.train_labeled_file_path = img_file_path[:791] + img_file_path2[:] + img_file_path3[:]
      self.train_anno_file_path = anno_file_path[:791] + anno_file_path2[:] + anno_file_path3[:]

      self.val_file_path = img_file_path[791:]
      self.val_anno_file_path = anno_file_path[791:]

      self.test_file_path = img_file_path4[:]
      self.test_anno_file_path = anno_file_path4[:]

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
  
##########　以降はDataset, DataTransformの定義　##########

# Chunデータセットの大画像1枚を分割したデータセット
class ChunPanoImageDataset(Dataset):
    def __init__(self, image_dir, file, transform=True):
        self.image_dir = image_dir
        self.transform = transform
        self.file = file
        
        # 画像ファイルと対応するセグメンテーションマスクのリストを作成
        self.image_files = glob.glob(self.image_dir + self.file + '*')
        self.image_files = sorted(self.image_files, key=lambda x: (int(x.split('_')[3]), int(x.split('_')[4].rstrip('.jpg'))))
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 画像の読み込み
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        
        resize = transforms.Resize((256,256))
        image = resize(image)
        
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
            
        return image
      
      
# ディレクトリに含まれる画像のみを返すデータセット
class OnlyImageDataset(Dataset):
    def __init__(self, image_dir, transform=True):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_files = glob.glob(self.image_dir + '*')
        self.image_files = sorted(self.image_files)
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 画像の読み込み
        img_name = self.image_files[idx]
        image = Image.open(img_name)
        
        resize = transforms.Resize((256,256))
        image = resize(image)
        
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
            
        return image

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
    

class PseudoLabeledDataset(Dataset):
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
    
    
class PseudoLabeledTransform():
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


class UnlabeledDataset(Dataset):
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

class UnlabeledTransform():
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