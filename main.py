# train, inference, save_maskを逐次的に実行する
import train, inference, save_mask

date = "231015"
folnames = [date+"_iter1", date+"_iter2", date+"_iter3", date+"_iter4", date+"_iter5"]
for i in range(5):
    if i==0:
        train(former_folname=None, folname=folnames[i])
        save_mask(folname=folnames[i])
    else:
        train(former_folname=folnames[i-1], folname=folnames[i])
        scores = inference(former_folname=folnames[i-1], folname=folnames[i])
        save_mask(folname=folnames[i])