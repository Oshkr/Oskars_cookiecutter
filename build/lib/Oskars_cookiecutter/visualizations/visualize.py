from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from visualizations.helper import *

def make_visualizations(tr_loss:list,
                        te_loss:list):
    
    print('saving plot...')

    epochs = np.linspace(0,2,len(tr_loss))
    
    fig,ax = plt.subplots()
    ax.plot(epochs,tr_loss,label = 'tr_loss')
    ax.plot(epochs,te_loss,label = 'te_loss')
    fig.legend()
    
    fig.savefig('reports/figures/loss_plot.png')

def visualize_images(imgs,predictions):

    for i in range(imgs.shape[0]):

        view_classify(imgs[i],predictions[i,:])
        plt.savefig(f'reports/figures/prediction_{i}')


if __name__=='__main__':
    visualize_images(1,2)
