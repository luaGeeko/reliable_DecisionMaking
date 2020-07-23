import copy
import load_data
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


def get_dummy_data_for_neuron_type(trial_features, number_of_neurons, neuron_type, number_of_time_bins):
    firings = np.zeros((number_of_neurons, trial_features.shape[1], number_of_time_bins))
    if neuron_type == 'random':
        # means = np.round(np.random.rand(trial_features.shape[1]) * number_of_time_bins, 0)  # random means
        sd = np.random.rand(trial_features.shape[1]) * 2  # random sd
        # firings = add sd to firings in the correct shape

    if neuron_type == 'right_choice':
        pass  # have a peak before the response if there is a right choice otherwise do not respond
    if neuron_type == 'left_choice':
        pass  # have a peak before the response if there is a left choice otherwise do not respond
    if neuron_type == 'peak_at_response':
        pass  # respond regardless of decision
    if neuron_type == 'ramp_to_action':
        pass  # increase / decrease activity until action regardless of decision

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
    trial_features[3] = response_times[0]
    return trial_features


def make_dummy_data_for_session(simulated_data, number_of_neurons, number_of_trials, number_of_time_bins):
    simulated_firing = np.zeros((number_of_neurons, number_of_trials, number_of_time_bins))
    neuron_types_added = []
    trial_feature_matrix = get_trial_feature_matrix(simulated_data)  # columns: left, right, no go, response_times
    types = ['right_choice', 'left_choice', 'peak_at_response', 'random', 'ramp_to_action']
    type_probabilities = [0.04, 0.06, 0.4, 0.3, 0.2]
    for index, neuron_type in enumerate(types):
        if index == len(types):
            number_of_neurons_to_generate = number_of_neurons - len(neuron_types_added)
        else:
            number_of_neurons_to_generate = int(number_of_neurons * type_probabilities[index])
        simulated_firing_neuron = get_dummy_data_for_neuron_type(trial_feature_matrix, number_of_neurons_to_generate, neuron_type, number_of_time_bins)
        # update simulated_firing with this

    return simulated_firing


def generate_dummy_data():
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