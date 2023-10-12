from utils.segmentation_eval import DiceScore, Accuracy, Precision, Recall, Specificity

n_samples = 100
eval_method = DiceScore()
eval_method1 = Accuracy()
eval_method2 = Specificity()
eval_method3 = Recall()
eval_method4 = Precision()

model.to(device)
model_path = 'weights/231010_weights/model_528.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
with torch.no_grad():
    scores = []
    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    for image,mask in test_dataloader:
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

print(torch.mean(scores))
print(torch.mean(scores1))
print(torch.mean(scores2))
print(torch.mean(scores3))
print(torch.mean(scores4))