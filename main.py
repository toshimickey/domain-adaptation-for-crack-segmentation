# train, inference, save_maskを逐次的に実行する
import train, inference, save_mask
import csv
from utils.module import write_to_csv

project = "231015"
folnames = [project+"_iter1", project+"_iter2", project+"_iter3", project+"_iter4", project+"_iter5"]
csv_filename = 'results/'+project+'_results/results.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Iteration', 'F1-Score', 'Accuracy', 'Specificity', 'Recall', 'Precision'])

for i in range(5):
    if i==0:
        train(former_folname="hoge", folname=folnames[i], first=True, net="deeplab", epochs=1000)
        scores = inference("hoge", folname=folnames[i], net="deeplab")
        write_to_csv(i+1, scores, csv_filename)
        save_mask(former_folname="hoge", folname=folnames[i], net="deeplab")
    else:
        train(former_folname=folnames[i-1], folname=folnames[i], first=False, net="deeplab", epochs=300)
        scores = inference(former_folname=folnames[i-1], folname=folnames[i], net="deeplab")
        write_to_csv(i+1, scores, csv_filename)
        save_mask(former_folname=folnames[i-1], folname=folnames[i], net="deeplab")