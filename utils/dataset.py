# define make_datapath_list
class make_datapath_list():
  def __init__(self):
    img_file_path = sorted(glob.glob('data/Conglomerate/Train/images/*'))
    anno_file_path = sorted(glob.glob('data/Conglomerate/Train/masks/*'))
    combined = list(zip(img_file_path, anno_file_path))
    random.shuffle(combined)
    img_file_path, anno_file_path = zip(*combined)

    img_file_path2 = sorted(glob.glob('data/Chundata/original_split/*'))
    anno_file_path2 = sorted(glob.glob('data/Chundata/teacher_split/*'))
    # combined2 = list(zip(img_file_path2, anno_file_path2))
    # random.shuffle(combined2)
    # img_file_path2, anno_file_path2 = zip(*combined2)

    # ここを変更しよう
    mean_file_path = sorted(glob.glob('data/Chundata_unlabeled_mask/231007_iter1/pred_mean_corrected/*'))
    var_file_path = sorted(glob.glob('data/Chundata_unlabeled_mask/231007_iter1/pred_var/*'))


    self.train_labeled_file_path = img_file_path[:7919]
    self.train_anno_file_path = anno_file_path[:7919]

    self.train_unlabeled_file_path = img_file_path2[:4292]
    self.train_unlabeled_mean_path = mean_file_path
    self.train_unlabeled_var_path = var_file_path

    self.val_file_path = img_file_path[7919:9899]
    self.val_anno_file_path = anno_file_path[7919:9899]

    self.test_file_path = img_file_path2[4292:5292]
    self.test_anno_file_path = anno_file_path2[4292:5292]

  def get_list(self, path_type):
    if path_type=="train_labeled":
      file_path = self.train_labeled_file_path
      anno_path = self.train_anno_file_path
    elif path_type=="train_unlabeled":
      file_path = self.train_unlabeled_file_path
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
      for path in anno_path:
        anno_list.append(path)
      return img_list, anno_list

makepath = make_datapath_list()
train_labeled_img_list, train_labeled_anno_list = makepath.get_list("train_labeled")
train_unlabeled_img_list, train_unlabeled_mean_list, train_unlabeled_var_list = makepath.get_list("train_unlabeled")
val_img_list, val_anno_list = makepath.get_list("val")

from torch.utils.data import Dataset
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
    
import torchvision.transforms as transforms
class LabeledTransform():
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.mean = [0.473, 0.493, 0.504]
        self.std = [0.163, 0.154, 0.153]

    def __call__(self, image, mask):
        # # ランダムなスケールを1.0~2.0の中で選択する
        # scale = torch.FloatTensor(1).uniform_(1.0, 2.0)

        # w, h = image.size
        # new_width = int(round(w * scale.item()))
        # new_height = int(round(h * scale.item()))

        # # 画像をスケール倍にリサイズする
        # image = transforms.Resize((new_width, new_height))(image)
        # mask = transforms.Resize((new_width, new_height))(mask)

        # ランダムクロップ
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

        # 水平反転
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # 垂直反転
        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # # -20~20のランダム回転
        # angle = random.randint(-20, 20)
        # image = transforms.functional.rotate(image,angle)
        # mask = transforms.functional.rotate(mask,angle)

        # # imageの標準化
        # normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # image = normalize(transforms.ToTensor()(image))

        # テンソルに変換
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        mask = torch.where(mask > 0.5, torch.tensor(1), torch.tensor(0))  # しきい値を超える場合は1、超えない場合は0に変換

        return image, mask
    
import torchvision.transforms as transforms
class ValLabeledTransform():
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.mean = [0.473, 0.493, 0.504]
        self.std = [0.163, 0.154, 0.153]

    def __call__(self, image, mask):
        # リサイズ
        #image = transforms.Resize(self.resize)(image)
        #mask = transforms.Resize(self.resize)(mask)

        # センタークロップ
        image = transforms.CenterCrop(self.crop_size)(image)
        mask = transforms.CenterCrop(self.crop_size)(mask)

        # # imageの標準化
        # normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # image = normalize(transforms.ToTensor()(image))

        # テンソルに変換
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        mask = torch.where(mask > 0.5, torch.tensor(1), torch.tensor(0))

        return image, mask
    
from torch.utils.data import Dataset
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
    
import torchvision.transforms as transforms
class UnlabeledTransform():
    def __init__(self, crop_size):
        self.crop_size = crop_size
        self.mean = [0.473, 0.493, 0.504]
        self.std = [0.163, 0.154, 0.153]

    def __call__(self, image, mean, var):
        # センタークロップ
        image = transforms.CenterCrop(self.crop_size)(image)

        # # ランダムなスケールを1.0~2.0の中で選択する
        # scale = torch.FloatTensor(1).uniform_(1.0, 2.0)

        # w, h = image.size
        # new_width = int(round(w * scale.item()))
        # new_height = int(round(h * scale.item()))

        # # 画像をスケール倍にリサイズする
        # image = transforms.Resize((new_width, new_height))(image)
        # mean = transforms.Resize((new_width, new_height))(mean)
        # var = transforms.Resize((new_width, new_height))(var)

        # # ランダムクロップ
        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
        # image = transforms.functional.crop(image, i, j, h, w)
        # mean = transforms.functional.crop(mean, i, j, h, w)
        # var = transforms.functional.crop(var, i, j, h, w)

        #####保存時はここをコメントアウトする#####
        # # 水平反転
        # if random.random() > 0.5:
        #     image = transforms.functional.hflip(image)
        #     mean = transforms.functional.hflip(mean)
        #     var = transforms.functional.hflip(var)

        # # 垂直反転
        # if random.random() > 0.5:
        #     image = transforms.functional.vflip(image)
        #     mean = transforms.functional.vflip(mean)
        #     var = transforms.functional.vflip(var)
        ###########コメントアウト###########

        # imageの標準化
        # normalize = transforms.Normalize(mean=self.mean, std=self.std)
        # image = normalize(transforms.ToTensor()(image))

        # テンソルに変換
        image = transforms.ToTensor()(image)
        mean = transforms.ToTensor()(mean)
        mean = torch.where(mean > 0.5, torch.tensor(1), torch.tensor(0))
        var = transforms.ToTensor()(var)

        return image, mean, var