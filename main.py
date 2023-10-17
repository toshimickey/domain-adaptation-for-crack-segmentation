# train, inference, save_maskを逐次的に実行する
from train import train
from inference import inference
from save_mask import save_mask
import csv, os
from utils.module import write_to_csv

project = "231015"
folnames = [project+"_iter1", project+"_iter2", project+"_iter3", project+"_iter4", project+"_iter5"]
os.makedirs('results/'+ project +'_results', exist_ok=True)
csv_filename = 'results/'+project+'_results/results.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'F1-Score', 'Accuracy', 'Specificity', 'Recall', 'Precision'])

for i in range(len(folnames)):
    if i==0:
        train(former_folname="hoge", folname=folnames[i], first=True, net="deeplab", epochs=1000, batch_size=64, num_workers=2)
        scores = inference("hoge", folname=folnames[i], net="deeplab", batch_size=64, num_workers=2)
        write_to_csv(i+1, scores, csv_filename)
        save_mask(former_folname="hoge", folname=folnames[i], net="deeplab", batch_size=64, num_workers=2)
    else:
        train(former_folname=folnames[i-1], folname=folnames[i], first=False, net="deeplab", epochs=300, batch_size=64, num_workers=2)
        scores = inference(former_folname=folnames[i-1], folname=folnames[i], net="deeplab", batch_size=64, num_workers=2)
        write_to_csv(i+1, scores, csv_filename)
        save_mask(former_folname=folnames[i-1], folname=folnames[i], net="deeplab", batch_size=64, num_workers=2)