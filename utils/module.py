import csv

class EarlyStopping:
    def __init__(self, patience=50, delta=0, mode='min'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif (self.mode == 'min' and val_loss < self.best_score - self.delta) or \
                (self.mode == 'max' and val_loss > self.best_score + self.delta):
            self.counter = 0
            self.best_score = val_loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def write_to_csv(epoch, loss_lst, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        loss_lst.insert(0,epoch)
        writer.writerow(loss_lst)