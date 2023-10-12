train_labeled_dataloader = data.DataLoader(
    train_labeled_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

train_unlabeled_dataloader = data.DataLoader(
    train_unlabeled_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)