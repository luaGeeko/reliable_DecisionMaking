from matplotlib import pyplot as plt
import numpy as np
import os


dirname = os.path.dirname(__file__)


# source: https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/load_steinmetz_decisions.ipynb#scrollTo=mmOarX5w16CR
def plot_population_average(dat, file_name='population_average'):
    dt = dat['bin_size']  # binning at 10 ms
    NT = dat['spks'].shape[-1]

    ax = plt.subplot(1,5,1)
    response = dat['response'] # right - nogo - left (-1, 0, 1)
    vis_right = dat['contrast_right'] # 0 - low - high
    vis_left = dat['contrast_left'] # 0 - low - high
    plt.plot(dt * np.arange(NT), 1/dt * dat['spks'][:,response>=0].mean(axis=(0,1))) # left responses
    plt.plot(dt * np.arange(NT), 1/dt * dat['spks'][:,response<0].mean(axis=(0,1))) # right responses
    plt.plot(dt * np.arange(NT), 1/dt * dat['spks'][:,vis_right>0].mean(axis=(0,1))) # stimulus on the right
    plt.plot(dt * np.arange(NT), 1/dt * dat['spks'][:,vis_right==0].mean(axis=(0,1))) # no stimulus on the right

    plt.legend(['left resp', 'right resp', 'right stim', 'no right stim'], fontsize=12)
    ax.set(xlabel  = 'time (sec)', ylabel = 'firing rate (Hz)')
    if not os.path.isdir(dirname + '/figures/'):
        os.mkdir(dirname + '/figures/')
    plt.savefig(dirname + '/figures/' + file_name + '.png')