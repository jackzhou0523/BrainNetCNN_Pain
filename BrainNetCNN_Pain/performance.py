import numpy as np
from matplotlib import pyplot as plt


def performance(k,foldperf):
    validl_f,tl_f,valida_f,ta_f=[],[],[],[]
    for f in range(1,k+1):

         tl_f.append(np.mean(foldperf['fold{}'.format(f)]['train_loss']))
         validl_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_loss']))

         ta_f.append(np.mean(foldperf['fold{}'.format(f)]['train_acc']))
         valida_f.append(np.mean(foldperf['fold{}'.format(f)]['valid_acc']))

    print('Performance of {} fold cross validation'.format(k))
    print("Average Training Loss: {:.3f} \t Average Valid Loss: {:.3f} \t Average Training Acc: {:.2f} \t "
          "Average Valid Acc: {:.2f}".format(np.mean(tl_f),np.mean(validl_f),np.mean(ta_f),np.mean(valida_f)))

def plot_results(num_epochs, foldperf,plot = 'loss'):
    diz_ep = {'train_loss_ep':[],'valid_loss_ep':[],'train_acc_ep':[],'valid_acc_ep':[]}

    for i in range(num_epochs):
          diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
          diz_ep['valid_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_loss'][i] for f in range(k)]))
          diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(k)]))
          diz_ep['valid_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_acc'][i] for f in range(k)]))

    if plot == 'loss':
        # Plot losses
        plt.figure(figsize=(10,8))
        plt.semilogy(diz_ep['train_loss_ep'], label='Train')
        plt.semilogy(diz_ep['valid_loss_ep'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.grid()
        plt.legend()
        plt.title('CNN loss')
        plt.show()
    elif plot == 'acc':
        # Plot accuracies
        plt.figure(figsize=(10,8))
        plt.semilogy(diz_ep['train_acc_ep'], label='Train')
        plt.semilogy(diz_ep['valid_acc_ep'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        #plt.grid()
        plt.legend()
        plt.title('CNN accuracy')
        plt.show()
    else:
        print('Error plotting!')