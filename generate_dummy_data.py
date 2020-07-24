import copy
import load_data
import math_utility
import numpy as np
import split_test_and_training_data


def get_training_data_for_simulation():
    """
    Load the training data. We need this to get the real behavioural data.
    We will later replace the real firing data with simulated data.
    """
    mouse_data = load_data.load_mouse_data()
    training_data = split_test_and_training_data.get_training_data(mouse_data)
    simulated_data = copy.deepcopy(training_data)
    return simulated_data


def make_random_neuron(trial_features, number_of_time_bins):
    number_of_trials = trial_features.shape[1]
    firings_on_trial = math_utility.get_mixture_of_random_gaussians(15, np.arange(number_of_time_bins))
    for trial in range(number_of_trials - 1):
        firings_on_trial = np.vstack((firings_on_trial, math_utility.get_mixture_of_random_gaussians(15, np.arange(number_of_time_bins))))

    return firings_on_trial


def make_right_choice_neuron(trial_features, number_of_time_bins):
    number_of_trials = trial_features.shape[1]
    earliest_response = trial_features[-1].min() * 100
    # setting a reasonable range for peak
    forward_time_shift_relative_to_response = np.random.uniform(2, earliest_response - 5)
    sigma = np.random.uniform(0, number_of_time_bins / 4)
    if trial_features[0][0] == 1:  # behaviour was a right choice
        mu = trial_features[-1][0] * 100 - forward_time_shift_relative_to_response
        firings_on_trial = math_utility.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
    else:
        firings_on_trial = np.zeros(number_of_time_bins)
    for trial in range(number_of_trials - 1):
        if trial_features[0][trial + 1] == 1:  # behaviour was a right choice
            mu = trial_features[-1][trial + 1] * 100 - forward_time_shift_relative_to_response
            next_firings = math_utility.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
        else:
            next_firings = np.zeros(number_of_time_bins)
        firings_on_trial = np.vstack((firings_on_trial, next_firings))
    return firings_on_trial


def make_left_choice_neuron(trial_features, number_of_time_bins):
    number_of_trials = trial_features.shape[1]
    earliest_response = trial_features[-1].min() * 100
    # setting a reasonable range for peak
    forward_time_shift_relative_to_response = np.random.uniform(2, earliest_response - 5)
    sigma = np.random.uniform(0, number_of_time_bins / 4)
    if trial_features[1][0] == 1:  # behaviour was a left choice
        mu = trial_features[-1][0] * 100 - forward_time_shift_relative_to_response
        firings_on_trial = math_utility.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
    else:
        firings_on_trial = np.zeros(number_of_time_bins)
    for trial in range(number_of_trials - 1):
        if trial_features[1][trial + 1] == 1: # behaviour was a left choice
            mu = trial_features[-1][trial + 1] * 100 - forward_time_shift_relative_to_response
            next_firings = math_utility.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
        else:
            next_firings = np.zeros(number_of_time_bins)
        firings_on_trial = np.vstack((firings_on_trial, next_firings))
    return firings_on_trial


# peak activity is relative to the response time but not behaviour
def make_response_neuron(trial_features, number_of_time_bins):
    number_of_trials = trial_features.shape[1]
    earliest_response = trial_features[-1].min() * 100
    # setting a reasonable range for peak
    forward_time_shift_relative_to_response = np.random.uniform(2, earliest_response - 5)
    sigma = np.random.uniform(0, number_of_time_bins / 4)
    mu = trial_features[-1][0] * 100 - forward_time_shift_relative_to_response
    firings_on_trial = math_utility.my_gaussian(np.arange(number_of_time_bins), mu, sigma)

    for trial in range(number_of_trials - 1):
        mu = trial_features[-1][trial + 1] * 100 - forward_time_shift_relative_to_response
        next_firings = math_utility.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
        firings_on_trial = np.vstack((firings_on_trial, next_firings))
    return firings_on_trial


def make_ramp_neuron(trial_features, number_of_time_bins):
    number_of_trials = trial_features.shape[1]
    # stimulus onset is at bin # 50
    mu = 51
    sigma = np.random.uniform(5, 10)
    # gauss = math_utility.my_gaussian(np.arange(number_of_time_bins), mu, sigma)
    linear = np.hstack((np.zeros(50), np.arange(50, number_of_time_bins)))
    linear /= np.sum(linear)
    slope = np.random.uniform(2, 6)
    firings_on_trial = linear * slope

    for trial in range(number_of_trials):
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
        if index == len(types):
            number_of_neurons_to_generate = number_of_neurons - len(neuron_types_added)

        else:
            number_of_neurons_to_generate = int(number_of_neurons * type_probabilities[index])
        neuron_counter += number_of_neurons_to_generate
        simulated_firing_neurons = get_dummy_data_for_neuron_type(trial_feature_matrix, number_of_neurons_to_generate,
                                                                 neuron_type, number_of_time_bins)
        simulated_firing[already_added:neuron_counter, :, :] = simulated_firing_neurons
        already_added += number_of_neurons_to_generate
        neuron_types_added.extend([neuron_type] * number_of_neurons_to_generate)
        print('maybe')

    return simulated_firing


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
        simulated_firing = make_dummy_data_for_session(simulated_data[session_id], number_of_neurons, number_of_trials, number_of_time_bins)
        simulated_data[session_id]['spks'] = simulated_firing  # overwrite real data with simulated data


def main():
    generate_dummy_data()


if __name__ == '__main__':
    main()