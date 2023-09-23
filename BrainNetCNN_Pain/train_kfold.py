import os

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import optim, nn
from torch.utils.data import SubsetRandomSampler, DataLoader

from cm import plot_confusion_matrix
from epoch import train_model, valid_model
from main import BrainNetCNN
from performance import performance
from read_dataset import train, test, categories
# from BrainNetCNN_pain.utils import reset_weights

model_dir = '../model/BrainNetCNN_pain/dict'
model_name = '2E2E_b8_k5_r42_leaky/'
save_file_name =os.path.join(model_dir,model_name)
if not os.path.exists(save_file_name):
    os.makedirs(save_file_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
num_epochs=250
max_epochs_stop = 100
print_every = 1

batch_size=8
k=5
kf=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}


dataset = train
test_loader = DataLoader(test, batch_size=batch_size)
# test_features, test_labels = next(iter(test_loader))
# model = BrainNetCNN(test_features)
# def init_weights_he(m):
#     #https://keras.io/initializers/#he_uniform
#     print(m)
#     if type(m) == torch.nn.Linear:
#         fan_in = model.dense1.in_features
#         he_lim = np.sqrt(6) / fan_in
#         m.weight.data.uniform_(-he_lim,he_lim)
#
# model.apply(init_weights_he)
# model.to(device)

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):

    print('Fold {}'.format(fold + 1))
    save_fold_name = save_file_name + 'fold_' + str(fold+1)


    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)


    train_features, train_labels = next(iter(train_loader))
    model = BrainNetCNN(train_features)

    def init_weights_he(m):
        # https://keras.io/initializers/#he_uniform
        print(m)
        if type(m) == torch.nn.Linear:
            fan_in = model.dense1.in_features
            he_lim = np.sqrt(6) / fan_in
            m.weight.data.uniform_(-he_lim, he_lim)


    model.apply(init_weights_he)
    model.to(device)

    # wd = 0.0005
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=wd)
    optimizer = optim.Adam(model.parameters())

    history = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': []}

    # Start Training
    epochs_no_improve = 0
    valid_loss_min = np.Inf
    train_loss_min = np.Inf

    for epoch in range(num_epochs):
        # Training loop
        train_loss, train_acc = train_model(model, criterion, optimizer,train_loader)
        # Validation loop
        valid_loss, valid_acc = valid_model(model, criterion, valid_loader)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        # Print training and validation results
        if (epoch + 1) % print_every == 0:
            print(
                f'\nEpoch: {epoch + 1} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
            )
            print(
                f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
            )


        # Save the model if validation loss decreases
        if valid_loss < valid_loss_min:
            # Save model
            torch.save(model.state_dict(), save_fold_name)

            epochs_no_improve = 0
            valid_loss_min = valid_loss
            valid_best_acc = valid_acc
            early_train_acc = train_acc
            early_train_loss = train_loss
            best_epoch = epoch
        else:
            continue
            # epochs_no_improve += 1
            # if epochs_no_improve >= max_epochs_stop:
            #     print(
            #         f'\nEarly Stopping!')
            #     break
    # Load the best state dict
    model.load_state_dict(torch.load(save_fold_name))

    # model.optimizer = optimizer
    print(
        f'\nBest epoch: {best_epoch + 1} with Validation loss: {valid_loss_min:.2f} and Validation acc: {100 * valid_best_acc:.2f}%'
    )
    print(
        f'\t\t Training loss: {early_train_loss:.2f} and Training acc: {100 * early_train_acc:.2f}%'
    )


    foldperf['fold{}'.format(fold + 1)] = history


# Average performance
performance(k, foldperf)


diz_ep = {'train_loss_ep':[],'valid_loss_ep':[],'train_acc_ep':[],'valid_acc_ep':[]}

for i in range(num_epochs):
      diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
      diz_ep['valid_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['valid_loss'][i] for f in range(k)]))
      diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(k)]))
      diz_ep['valid_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['valid_acc'][i] for f in range(k)]))

# # model save on gpu
# model.eval()
# save_model_path = os.path.join(os.path.dirname(model_dir),'gpu',f'{os.path.dirname(model_name)}_{early_train_acc:.2f}_{valid_best_acc:.2f}.pt')
# torch.save(model,save_model_path)
#
# # save on cpu
# device = torch.device('cpu')
# cpu_PATH = os.path.join(os.path.dirname(model_dir),'cpu',f'{os.path.dirname(model_name)}_{early_train_acc:.2f}_{valid_best_acc:.2f}.pt')
# test_features, test_labels = next(iter(test_loader))
# model_cpu = BrainNetCNN(test_features)
# model_cpu.load_state_dict(torch.load(save_fold_name, map_location=device))
# torch.save(model_cpu,cpu_PATH)

#cm

from sklearn.metrics import confusion_matrix
from cm import plot_confusion_matrix
y_pred=[]
y_true=[]
for data, target in test_loader:
    data, target = data.to(device=device, dtype=torch.float), target.to(device=device)

    output = model(data)

    _, preds = torch.max(output, dim=1)

    y_pred.extend(preds.view(-1).detach().cpu().numpy())
    y_true.extend(target.view(-1).detach().cpu().numpy())

cmatrix = confusion_matrix(y_true, y_pred)
target_names = categories
print (cmatrix)


save_path_cm = 'C:/Users/Jack Zhou/Desktop/cm'
plot_confusion_matrix(cmatrix,
                          target_names,
                          save_path_cm = save_path_cm,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True)


# torch.save(model, 'k_cross_CNN.pt')
# for (data, target) in train_set:
#     imgs, targets = train_set[i]
#     print(imgs.shape)
#     print(targets)
