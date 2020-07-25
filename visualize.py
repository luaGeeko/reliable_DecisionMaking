from matplotlib import pyplot as plt
import math_utility
import numpy as np
import os
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d


dirname = os.path.dirname(__file__)

# source: https://colab.research.google.com/github/NeuromatchAcademy/course-content/blob/master/projects/load_steinmetz_decisions.ipynb#scrollTo=mmOarX5w16CR
def plot_population_average(dat, file_name='population_average'):
    dt = dat['bin_size']  # binning at 10 ms
    NT = dat['spks'].shape[-1]

    ax = plt.subplot(1,5,1)
    response = dat['response'] # right - nogo - left (-1, 0, 1)
    vis_right = dat['contrast_right'] # 0 - low - high
    vis_left = dat['contrast_left'] # 0 - low - high
    plt.plot(dt * np.arange(NT), 1/dt * np.nanmean(dat['spks'][:, response >= 0], axis=(0, 1)))  # left responses
    plt.plot(dt * np.arange(NT), 1/dt * np.nanmean(dat['spks'][:, response < 0], axis=(0, 1)))  # right responses
    plt.plot(dt * np.arange(NT), 1/dt * np.nanmean(dat['spks'][:, vis_right > 0], axis=(0, 1)))  # stimulus on the right
    plt.plot(dt * np.arange(NT), 1/dt * np.nanmean(dat['spks'][:, vis_right == 0], axis=(0, 1)))  # no stimulus on the right

    plt.legend(['left resp', 'right resp', 'right stim', 'no right stim'], fontsize=12)
    ax.set(xlabel  = 'time (sec)', ylabel = 'firing rate (Hz)')
    if not os.path.isdir(dirname + '/figures/'):
        os.mkdir(dirname + '/figures/')
    plt.savefig(dirname + '/figures/' + file_name + '.png')
    plt.close()


def plot_random_examples(data, file_name='random_examples', number_of_cells_to_plot=15):
    plt.figure()
    random_indices = np.random.randint(0, data['spks'].shape[0], size=number_of_cells_to_plot)  # randomize neurons
    for cell in random_indices:
        random_trial = np.random.randint(0, data['spks'].shape[1])  # randomize a trial
        plt.plot(data['spks'][cell, random_trial, :])
        plt.xlabel('Time (10 ms bins)')
        plt.ylabel('Normalized firing rate')
        plt.title('Random example neurons')

    plt.savefig(dirname + '/figures/' + file_name + '.png')
    plt.close()


def plot_example_from_each_neuron_type(data, file_name='type_examples'):
    types = np.unique(data['dummy_type'])
    for dummy_type in types:
        type_boolean_mask = pd.Series(data['dummy_type']) == dummy_type
        index_example_neuron = np.where(type_boolean_mask == True)[0][0]
        right_choice = data['response'] == -1
        example_right_trial = np.where(right_choice == True)[0][0]
        left_choice = data['response'] == 1
        example_left_trial = np.where(left_choice == True)[0][0]
        data_from_right_trial = data['spks'][index_example_neuron, example_right_trial, :]
        data_from_right_trial_filtered = math_utility.moving_average(data_from_right_trial, window=20)
        data_from_left_trial = data['spks'][index_example_neuron, example_left_trial, :]
        data_from_left_trial_filtered = math_utility.moving_average(data_from_left_trial, window=20)

        plt.figure()
        plt.xlabel('Time (10 ms bins)', fontsize=16)
        plt.ylabel('Firing rate', fontsize=16)
        plt.title('Activity of "' + dummy_type + '" neuron', fontsize=16)
        plt.plot(data_from_left_trial, color='black', linewidth=1, alpha=0.7, label='Left choice')
        plt.plot(data_from_right_trial, color='red', linewidth=1, alpha=0.7, label='Right choice')
        plt.plot(data_from_right_trial_filtered, color='red', linewidth=5)
        plt.plot(data_from_left_trial_filtered, color='black', linewidth=5)
        plt.legend(loc="upper right", frameon=False)

        plt.savefig(dirname + '/figures/' + file_name + '_' + dummy_type + '.png')
        plt.close()




