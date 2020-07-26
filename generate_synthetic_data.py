""" This module generates synthetic dataset for analysis and debugging purposed for our model """

import copy
from load_data import load_mouse_data
import utils
import numpy as np
import coloredlogs, logging
import os
from split_test_and_training_data import create_dataset_split_indices, get_training_data, train_test_split, get_test_data
import matplotlib.pyplot as plt
import timeit


# debugging
logger = logging.getLogger("generate_synthetic_data")
coloredlogs.install(level='INFO', logger=logger)

# random seed for reproducibility of experiements with dataset splits
np.random.seed(75)

def get_training_data_for_simulation():
    """
    load the training data. We need this to get the real behavioural data.
    We will later replace the real firing data with simulated data.
    """
    #mouse_data = load_mouse_data()
    training_data, _ = train_test_split()
    simulated_data = copy.deepcopy(training_data)
    return simulated_data


def make_random_neuron(trial_features, number_of_time_bins):
    """ 
    create random neuron with given number of trials and number of time bins 
    args:
        trial_features (numpy array) : trail features [left_response/right, nogo and response times] 
        number_of_time_bins (int) : number of time bins
    returns:
        numpy array : a neuron created with random values with given number of trials and number of time bins
    """
    # features X num_of_trials
    number_of_trials = trial_features.shape[1]
    temp_trials_created = []
    # creating random neuron with num_of_trials X random gaussion
    for _ in range(number_of_trials):
        temp_firings_on_trial = utils.get_mixture_of_random_gaussians(15, np.arange(number_of_time_bins))
        temp_trials_created.append(temp_firings_on_trial)
    firings_on_trial = np.asarray(temp_trials_created)
    return firings_on_trial


def make_right_choice_neuron(trial_features, number_of_time_bins):
    """ 
    create neuron simulating behaviour for right choice, given number of trials and number of time bins 
    args:
        trial_features (numpy array) : trail features [left_response/right, nogo and response times] 
        number_of_time_bins (int) : number of time bins
    returns:
        numpy array : a neuron with behaviour for right choice given number of trials and number of time bins
    """
    number_of_trials = trial_features.shape[1]
    earliest_response = trial_features[-1].min() * 100
    # setting a reasonable range for peak
    forward_time_shift_relative_to_response = np.random.uniform(2, earliest_response - 5)
    sigma = np.random.uniform(0, number_of_time_bins / 4)
    temp_right_choice = []
    for trial in range(number_of_trials):
        if trial_features[0][trial] == 1:  # behaviour was a right choice
            mu = trial_features[-1][trial] * 100 - forward_time_shift_relative_to_response
            next_firings = utils.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
        else:
            next_firings = np.zeros(number_of_time_bins)
        temp_right_choice.append(next_firings)
    
    firings_on_trial = np.asarray(temp_right_choice)
    return firings_on_trial

def make_left_choice_neuron(trial_features, number_of_time_bins):
    """ 
    create neuron simulating behaviour for left choice, given number of trials and number of time bins 
    args:
        trial_features (numpy array) : trail features [left_response/right, nogo and response times] 
        number_of_time_bins (int) : number of time bins
    returns:
        numpy array : a neuron with behaviour for left choice given number of trials and number of time bins
    """
    number_of_trials = trial_features.shape[1]
    earliest_response = trial_features[-1].min() * 100
    # setting a reasonable range for peak
    forward_time_shift_relative_to_response = np.random.uniform(2, earliest_response - 5)
    sigma = np.random.uniform(0, number_of_time_bins / 4)
    temp_left_choice = [] 
    for trial in range(number_of_trials):
        # behaviour was a left choice
        if trial_features[1][trial] == 1: 
            mu = trial_features[-1][trial] * 100 - forward_time_shift_relative_to_response
            next_firings = utils.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
        else:
            next_firings = np.zeros(number_of_time_bins)
        temp_left_choice.append(next_firings)

    firings_on_trial = np.asarray(temp_left_choice)
    return firings_on_trial


# peak activity is relative to the response time but not behaviour
def make_response_neuron(trial_features, number_of_time_bins):
    number_of_trials = trial_features.shape[1]
    earliest_response = trial_features[-1].min() * 100
    # setting a reasonable range for peak
    forward_time_shift_relative_to_response = np.random.uniform(2, earliest_response - 5)
    sigma = np.random.uniform(0, number_of_time_bins / 4)
    mu = trial_features[-1][0] * 100 - forward_time_shift_relative_to_response
    firings_on_trial = utils.my_gaussian(np.arange(number_of_time_bins), mu, sigma)

    for trial in range(number_of_trials - 1):
        mu = trial_features[-1][trial + 1] * 100 - forward_time_shift_relative_to_response
        next_firings = utils.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
        firings_on_trial = np.vstack((firings_on_trial, next_firings))
    return firings_on_trial


def make_ramp_neuron(trial_features, number_of_time_bins):
    number_of_trials = trial_features.shape[1]
    # stimulus onset is at bin # 50
    mu = 51
    sigma = np.random.uniform(5, 10)
    # gauss = utils.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
    linear = np.hstack((np.zeros(50), np.arange(50, number_of_time_bins)))
    linear /= np.sum(linear)
    slope = np.random.uniform(2, 6)
    firings_on_trial = linear * slope

    for trial in range(number_of_trials - 1):
        next_ramp = linear * slope
        firings_on_trial = np.vstack((firings_on_trial, next_ramp))
    return firings_on_trial


def get_dummy_data_for_neuron_type(trial_features, number_of_neurons, neuron_type, number_of_time_bins):
    firings = np.zeros((number_of_neurons, trial_features.shape[1], number_of_time_bins))
    if neuron_type == 'random':
        for neuron in range(number_of_neurons):
            random_neuron = make_random_neuron(trial_features, number_of_time_bins)
            firings[neuron, :, :] = random_neuron

    if neuron_type == 'right_choice':
        for neuron in range(number_of_neurons):
            right_choice_neuron = make_right_choice_neuron(trial_features, number_of_time_bins)
            firings[neuron, :, :] = right_choice_neuron
    if neuron_type == 'left_choice':
        for neuron in range(number_of_neurons):
            left_choice_neuron = make_left_choice_neuron(trial_features, number_of_time_bins)
            firings[neuron, :, :] = left_choice_neuron
    if neuron_type == 'peak_at_response':
        for neuron in range(number_of_neurons):
            response_neuron = make_response_neuron(trial_features, number_of_time_bins)
            firings[neuron, :, :] = response_neuron
    if neuron_type == 'ramp_to_action':
        for neuron in range(number_of_neurons):
            ramp_neuron = make_ramp_neuron(trial_features, number_of_time_bins)
            firings[neuron, :, :] = ramp_neuron

    return firings


def get_trial_feature_matrix(simulated_data):
    response_times = simulated_data['response_time']
    responses = simulated_data['response']
    right_mask = responses == -1
    left_mask = responses == 1
    no_go_mask = responses == 0

    number_of_trials = simulated_data['spks'].shape[1]
    # features are response_left, response_right, no_response
    trial_features = np.zeros((4, number_of_trials))
    trial_features[0] = right_mask
    trial_features[1] = left_mask
    trial_features[2] = no_go_mask
    trial_features[3] = response_times.flatten()
    return trial_features


def make_dummy_data_for_session(simulated_data, number_of_neurons, number_of_trials, number_of_time_bins):
    simulated_firing = np.zeros((number_of_neurons, number_of_trials, number_of_time_bins))
    neuron_types_added = []
    trial_feature_matrix = get_trial_feature_matrix(simulated_data)  # columns: left, right, no go, response_times
    types = ['right_choice', 'left_choice', 'peak_at_response', 'random', 'ramp_to_action']
    print('Neuron types generated: ' + str(types))
    type_probabilities = [0.04, 0.06, 0.4, 0.3, 0.2]
    print('With occurance probabilities: ' + str(type_probabilities))
    neuron_counter = 0
    already_added = 0
    for index, neuron_type in enumerate(types):
        if neuron_type == types[-1]:
            number_of_neurons_to_generate = number_of_neurons - len(neuron_types_added)

        else:
            number_of_neurons_to_generate = int(number_of_neurons * type_probabilities[index])
        neuron_counter += number_of_neurons_to_generate
        simulated_firing_neurons = get_dummy_data_for_neuron_type(trial_feature_matrix, number_of_neurons_to_generate,
                                                                 neuron_type, number_of_time_bins)
        simulated_firing[already_added:neuron_counter, :, :] = simulated_firing_neurons
        already_added += simulated_firing_neurons.shape[0]
        neuron_types_added.extend([neuron_type] * simulated_firing_neurons.shape[0])

    return simulated_firing, neuron_types_added


def save_dummy_data(data, file_name):
    dirname = os.path.dirname(__file__)
    if not os.path.isdir(dirname + '/dummy_data/'):
        os.mkdir(dirname + '/dummy_data/')
    np.save(dirname + '/dummy_data/' + file_name, [data])


def generate_dummy_data():
    print('Generating dummy data with multiple neuron types.')
    # load data (to get behavioural variables and shapes)
    simulated_data = get_training_data_for_simulation()  # I will update the real firing with simulated.
    # iterate over sessions
    for session_id in range(len(simulated_data)):
        trials_in_session = simulated_data[session_id]['spks']  # firing from session
        number_of_neurons = trials_in_session.shape[0]
        number_of_trials = trials_in_session.shape[1]
        number_of_time_bins = trials_in_session.shape[2]
        print('Make dummy data for this session: session #' + str(session_id))
        simulated_firing, neuron_types_added = make_dummy_data_for_session(simulated_data[session_id], number_of_neurons, number_of_trials, number_of_time_bins)
        noise = np.random.normal(0, .01, simulated_firing.shape)
        simulated_firing += noise
        simulated_data[session_id]['spks'] = simulated_firing  # overwrite real data with simulated data
        simulated_data[session_id]['dummy_type'] = neuron_types_added
        save_dummy_data(simulated_data[session_id], str(session_id))
    return simulated_data


data = load_mouse_data()
session_id = 2
feature_matrix = get_trial_feature_matrix(data[session_id])
choice = make_right_choice_neuron(feature_matrix, number_of_time_bins=250)
time_checker = utils.time_cal(make_right_choice_neuron, feature_matrix, number_of_time_bins=250)
print (timeit.timeit(time_checker, number=100))