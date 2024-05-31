from os.path import join

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from lib.test_metrics import *
from lib.utils import to_numpy, get_range
import itertools

plt.rcParams['savefig.dpi'] = 300


def set_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def compare_hists(x_real, x_fake, ax=None, log=False, label=None):
    """ Computes histograms and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if label is not None:
        label_historical = 'Historical ' + label
        label_generated = 'Generated ' + label
    else:
        label_historical = 'Historical'
        label_generated = 'Generated'
    bin_edges = ax.hist(x_real.flatten(), bins=80, alpha=0.6, density=True, label=label_historical)[1]
    ax.hist(x_fake.flatten(), bins=bin_edges, alpha=0.6, density=True, label=label_generated)
    ax.grid()
    set_style(ax)
    ax.legend()
    if log:
        ax.set_ylabel('log-pdf')
        ax.set_yscale('log')
    else:
        ax.set_ylabel('pdf')
    return ax


def compare_acf(x_real, x_fake, ax=None, max_lag=64, CI=True, dim=(0, 1), drop_first_n_lags=0):
    """ Computes ACF of historical and (mean)-ACF of generated and plots those. """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    acf_real_list = cacf_torch(x_real, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_real = np.mean(acf_real_list, axis=0)

    acf_fake_list = cacf_torch(x_fake, max_lag=max_lag, dim=dim).cpu().numpy()
    acf_fake = np.mean(acf_fake_list, axis=0)

    ax.plot(acf_real[drop_first_n_lags:], label='Historical', linewidth=1)
    ax.plot(acf_fake[drop_first_n_lags:], label='Generated', alpha=0.8, linewidth=1)

    if CI:
        acf_fake_std = np.std(acf_fake_list, axis=0)
        ub = acf_fake + acf_fake_std
        lb = acf_fake - acf_fake_std

        for i in range(acf_real.shape[-1]):
            ax.fill_between(
                range(acf_fake[:, i].shape[0]),
                ub[:, i], lb[:, i],
                color='orange',
                alpha=.3
            )
    set_style(ax)
    ax.set_xlabel('Lags')
    ax.set_ylabel('ACF')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True)
    ax.legend()
    return ax

# Below code from https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/visualization_metrics.py
def visualization (ori_data, generated_data, analysis):
  """Using PCA or tSNE for generated and original data visualization.
  
  Args:
    - ori_data: original data
    - generated_data: generated synthetic data
    - analysis: tsne or pca
  """  
  from sklearn.manifold import TSNE
  from sklearn.decomposition import PCA
  # Analysis sample size (for faster computation)
  #anal_sample_no = min([1000, len(ori_data)])
  anal_sample_no = len(ori_data)
  #idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  
  
  #ori_data = ori_data[idx]
  #generated_data = generated_data[idx]
  
  no, seq_len, dim = ori_data.shape  
  
  for i in range(anal_sample_no):
    if (i == 0):
      prep_data = np.reshape(np.mean(ori_data[0,:,:], 1), [1,seq_len])
      prep_data_hat = np.reshape(np.mean(generated_data[0,:,:],1), [1,seq_len])
    else:
      prep_data = np.concatenate((prep_data, 
                                  np.reshape(np.mean(ori_data[i,:,:],1), [1,seq_len])))
      prep_data_hat = np.concatenate((prep_data_hat, 
                                      np.reshape(np.mean(generated_data[i,:,:],1), [1,seq_len]))) 
  # Visualization parameter        
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    
    

  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(prep_data)
    pca_results = pca.transform(prep_data)
    pca_hat_results = pca.transform(prep_data_hat)
    
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2,label = "Synthetic")
  
    ax.legend()  
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    
  elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis = 0)
    
    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(prep_data_final)
      
    # Plotting
    f, ax = plt.subplots(1)
    
    plt.scatter(tsne_results[:anal_sample_no,0], tsne_results[:anal_sample_no,1], 
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results[anal_sample_no:,0], tsne_results[anal_sample_no:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")
  
    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    

def plot_summary(x_fake, x_real, max_lag=None, labels=None):
    if max_lag is None:
        max_lag = min(128, x_fake.shape[1])

    from lib.test_metrics import skew_torch, kurtosis_torch
    dim = x_real.shape[2]
    _, axes = plt.subplots(dim, 3, figsize=(25, dim * 5))

    if len(axes.shape) == 1:
        axes = axes[None, ...]
    for i in range(dim):
        x_real_i = x_real[..., i:i + 1]
        x_fake_i = x_fake[..., i:i + 1]

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 0])

        def text_box(x, height, title):
            textstr = '\n'.join((
                r'%s' % (title,),
                # t'abs_metric=%.2f' % abs_metric
                r'$s=%.2f$' % (skew_torch(x).item(),),
                r'$\kappa=%.2f$' % (kurtosis_torch(x).item(),))
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            axes[i, 0].text(
                0.05, height, textstr,
                transform=axes[i, 0].transAxes,
                fontsize=14,
                verticalalignment='top',
                bbox=props
            )

        text_box(x_real_i, 0.95, 'Historical')
        text_box(x_fake_i, 0.70, 'Generated')

        compare_hists(x_real=to_numpy(x_real_i), x_fake=to_numpy(x_fake_i), ax=axes[i, 1], log=True)
        compare_acf(x_real=x_real_i, x_fake=x_fake_i, ax=axes[i, 2], max_lag=max_lag, CI=False, dim=(0, 1))


def compare_cross_corr(x_real, x_fake):
    """ Computes cross correlation matrices of x_real and x_fake and plots them. """
    x_real = x_real.reshape(-1, x_real.shape[2])
    x_fake = x_fake.reshape(-1, x_fake.shape[2])
    cc_real = np.corrcoef(to_numpy(x_real).T)
    cc_fake = np.corrcoef(to_numpy(x_fake).T)

    vmin = min(cc_fake.min(), cc_real.min())
    vmax = max(cc_fake.max(), cc_real.max())

    fig, axes = plt.subplots(1, 2)
    axes[0].matshow(cc_real, vmin=vmin, vmax=vmax)
    im = axes[1].matshow(cc_fake, vmin=vmin, vmax=vmax)

    axes[0].set_title('Real')
    axes[1].set_title('Generated')
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)


def plot_signature(signature_tensor, alpha=0.2):
    plt.plot(to_numpy(signature_tensor).T, alpha=alpha, linestyle='None', marker='o')
    plt.grid()


def savefig(filename, directory):
    plt.savefig(join(directory, filename))
    plt.close()


def create_summary(dataset, device, G, lags_past, steps, x_real, one=False):
    with torch.no_grad():
        x_past = x_real[:, :lags_past]
        if dataset in ['STOCKS', 'ECG']:
            x_p = x_past.clone().repeat(5, 1, 1)
        else:
            x_p = x_past.clone()
        if one:
            x_p = x_p[:1]
        x_fake_future = G.sample(steps, x_p.to(device))
    return x_fake_future

def plot_all(x_fake, x_real, counts,labels, dir):
    x_fake = x_fake.cpu().numpy()[0]
    x_real = x_real.cpu().numpy()[0]
    for count in counts:
        x_fake_tmp = x_fake[:count]
        x_real_tmp = x_real[:count]
        plot_fake_and_real(x_fake_tmp, x_real_tmp,labels, dir)
    
def plot_fake_and_real(x_fake, x_real,labels, dir):
    ymins_fake, ymaxs_fake = get_range(x_fake)
    ymins_real, ymaxs_real = get_range(x_real)
    ymins = np.vstack((ymins_fake, ymins_real)).min(axis=0)
    ymaxs = np.vstack((ymaxs_fake, ymaxs_real)).max(axis=0)
    c = len(x_fake)
    linewidth=0.4 if c <= 2000 else 0.2
    
    # fake
    plt.plot(x_fake, linewidth=linewidth)
    plt.ylim(ymins.min(), ymaxs.max())
    savefig(f'path_{c}_fake.png', dir)
    
    plt.plot(x_fake, linewidth=linewidth)
    savefig(f'path_{c}_fake_scaled.png', dir)
    
    create_subplots(x_fake, ymins, ymaxs, labels)
    savefig(f'path_{c}_subplots_fake.png', dir)
    
    #real
    plt.plot(x_real, linewidth=linewidth)
    plt.ylim(ymins.min(), ymaxs.max())
    savefig(f'path_{c}_real.png', dir)
    
    plt.plot(x_real, linewidth=linewidth)
    savefig(f'path_{c}_real_scaled.png', dir)
    
    create_subplots(x_real, ymins, ymaxs, labels)
    savefig(f'path_{c}_subplots_real.png', dir)
    
    plt.close()


def create_subplots(data, ymins, ymaxs, labels):
    n, m = data.shape
    linewidth = 0.5 if n >= 1000 else 1.0
    # Create a figure with m subplots, one for each column
    fig, axes = plt.subplots(m, 1, figsize=(10, 2 * m), sharex=True, sharey=False)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cycler = itertools.cycle(color_cycle)
    # If there's only one column, axes is not a list
    if m == 1:
        axes = [axes]

    # Plot each column in a separate subplot
    for i in range(m):
        axes[i].plot(data[:, i], color=next(cycler), linewidth=linewidth, label=labels[i])
        axes[i].set_ylim(ymins[i], ymaxs[i])
        axes[i].legend()

    plt.tight_layout()