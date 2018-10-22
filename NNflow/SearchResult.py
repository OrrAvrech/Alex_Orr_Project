import os
import numpy as np

from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
# from skopt.plots import plot_histogram, plot_objective_2D #functions changed in new version of skopt
from skopt.plots import plot_evaluations, plot_objective
import pickle
import matplotlib.pyplot as plt

def plotLearning(cfg, search_result):
    # hyper_params = {'data_params': data_params, 'learning_rate': learning_rate, 'num_conv_Bulks': num_conv_Bulks,
    #                 'kernel_size': kernel_size, 'activation': activation}

    # Plot and Save plot_convergence
    fig_converge = plot_convergence(search_result)
    fig_converge.figure.savefig(os.path.join(cfg.paths.learning, 'fig_converge.png'))

    # Save all hyperopt parameters and scores

    x_iters = [sublist[1:] for sublist in search_result.x_iters] # remove cfg class object from each
    print(x_iters)
    params_scores = sorted(zip(search_result.func_vals, x_iters))
    with open(os.path.join(cfg.paths.learning, 'params_scores.h5'), 'wb') as params_File:
        pickle.dump(params_scores, params_File, protocol=4)

    # Plot and Save activation histogram
    # fig_activation, _ = plot_evaluations(result=search_result, dimensions='activation')
    # fig_activation.savefig(os.path.join(cfg.paths.learning, 'fig_activation.png'))
    #
    # # Plot and Save kernel histogram
    # fig_kernel, _ = plot_evaluations(result=search_result, dimensions='kernel_size')
    # fig_kernel.savefig(os.path.join(cfg.paths.learning, 'fig_kernel.png'))

    # learning_Rate - num_conv_bulks : 2D Plot
    # fig_lr_ncb = plot_objective(result=search_result,
    #                             dimensions=['learning_rate', 'num_conv_Bulks'],
    #                             levels=50)
    # fig_lr_ncb.savefig(os.path.join(cfg.paths.learning, 'fig_lr_ncb.png'))
    #
    # # learning_Rate - kernel_size : 2D Plot
    # fig_lr_ks = plot_objective(result=search_result,
    #                            dimensions=['learning_rate', 'kernel_size'],
    #                                levels=50)
    # fig_lr_ks.savefig(os.path.join(cfg.paths.learning, 'fig_lr_ks.png'))
    #
    # # Dimensions Dependencies
    # dim_names = ['learning_rate', 'num_conv_Bulks', 'kernel_size']
    # # learning_Rate - num_conv_bulks - kernel_size : 3D Plot
    # fig_partial, _ = plot_objective(result=search_result, dimensions=dim_names)
    # fig_partial.savefig(os.path.join(cfg.paths.learning, 'fig_partial.png'))
    # # learning_Rate - num_conv_bulks - kernel_size : eval
    # fig_eval, _ = plot_evaluations(result=search_result, dimensions=dim_names)
    # fig_eval.savefig(os.path.join(cfg.paths.learning, 'fig_eval.png'))
    
def plot_activation_hist(samples):
    activations = []
    for s in samples:
        act = s[1][-1]
        activations.append(act)
    act_arr = np.array(activations)
    plt.hist(act_arr)
    plt.ylabel('numCounts')
    plt.xlabel('Activation')
    plt.show()
    
def plot_kernel_hist(samples):
    kernels = []
    for s in samples:
        kernel = s[1][-2]
        kernels.append(kernel)
    ker_arr = np.array(kernels)
    plt.hist(ker_arr)
    plt.ylabel('numCounts')
    plt.xlabel('Kernel Size')
    plt.show()
    
    
from scipy.interpolate import interp2d

#%%

with open(os.path.join('search_results', 'params_scores.h5'), 'rb') as paramsFile:
        search_results = pickle.load(paramsFile)

learning_rate = []
numConvBulks  = []
losses = []
for s in search_results:
    lr = s[1][2]
    conv = s[1][1]
    loss = s[0]
    learning_rate.append(lr)
    numConvBulks.append(conv)
    losses.append(loss)
lr_arr = np.array(learning_rate)
conv_arr = np.array(numConvBulks)
loss_arr = np.array(losses)

f = interp2d(lr_arr,conv_arr,loss_arr,kind="linear")

Z = f(lr_arr,conv_arr)

plt.imshow(Z, extent=[3, 11, 1, 5])
#plt.axis('off')
plt.show()
    
    
    
    