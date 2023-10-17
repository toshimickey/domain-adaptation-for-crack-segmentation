from utils.segmentation_eval import DiceScore, Accuracy, Precision, Recall, Specificity
import torch
import torch.utils.data as data
from dataloader.dataset import make_datapath_list, LabeledDataset, ValLabeledTransform
from models.bayesian_deeplab import DeepLabv3plusModel
from models.bayesian_unet import Unet256
from tqdm import tqdm

def inference(former_folname, folname, net="deeplab", batch_size=64, num_workers=2):
    makepath = make_datapath_list(former_folname, first=True)
    test_img_list, test_anno_list = makepath.get_list("test")
    test_dataset = LabeledDataset(test_img_list, test_anno_list, transform=ValLabeledTransform(crop_size=512))
    test_dataloader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    if net == "deeplab":
        model_wrapper = DeepLabv3plusModel()
        model = model_wrapper.get_model()
    else:
        model = Unet256((3, 512, 512))

    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    n_samples = 100
    eval_method = DiceScore()
    eval_method1 = Accuracy()
    eval_method2 = Specificity()
    eval_method3 = Recall()
    eval_method4 = Precision()

    model.to(device)
    model_path = f'weights/{folname}_weights/best.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        scores = []
        scores1 = []
        scores2 = []
        scores3 = []
        scores4 = []
        for image,mask in tqdm(test_dataloader):
            image = image.to(device,dtype=torch.float)
            mask = mask.to(device,dtype=torch.float)

            # sampling for n times
            preds = []
            for _ in range(n_samples):
                pred = model.forward(image).unsqueeze(0)
                preds.append(pred)
            assert not torch.equal(preds[0], preds[1])
            preds = torch.cat(preds)
            pred_mask = torch.mean(preds, dim=0) # (16,1,256,256)

            #　この中にsigmoid入っている
            scores.append(eval_method.forward(pred_mask,mask))
            scores1.append(eval_method1.forward(pred_mask,mask))
            scores2.append(eval_method2.forward(pred_mask,mask))
            scores3.append(eval_method3.forward(pred_mask,mask))
            scores4.append(eval_method4.forward(pred_mask,mask))

    scores = torch.tensor(scores)
    scores1 = torch.tensor(scores1)
    scores2 = torch.tensor(scores2)
    scores3 = torch.tensor(scores3)
    scores4 = torch.tensor(scores4)

    f1score = torch.mean(scores).item()
    accuracy = torch.mean(scores1).item()
    specificity = torch.mean(scores2).item()
    recall = torch.mean(scores3).item()
    precision = torch.mean(scores4).item()

    print(f'F1-Score : {f1score}')
    print(f'Accuracy : {accuracy}')
    print(f'Specificity : {specificity}')
    print(f'Recall : {recall}')
    print(f'Precision : {precision}')

    return [f1score, accuracy, specificity, recall, precision]
