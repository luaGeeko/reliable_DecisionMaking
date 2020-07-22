import load_data
import numpy as np


def make_binary_array_with_trial_types(number_of_trials, test_percentage):
    trial_type = np.zeros(number_of_trials)
    number_of_test_rows = int(round(number_of_trials * test_percentage / 100, 0))
    random_indices = np.random.choice(number_of_trials, number_of_test_rows, replace=False)
    trial_type[random_indices] = 1
    return trial_type


def label_test_and_training_data(all_data, test_percentage=20):
    """
    Adds a column 'test_data' to each trial with a binary array where 1 = the trial is test data, 0 = the trial
    is training data. For each neuron, it will classify 'test_percentage' proportion of trials as test trials.

    Args:
        all_data (class) : all simulated or mouse data
        test_percentage (float) : the percentage of trials per neuron that will be test trials
                                   0 >= test_percentage <= 1
    Returns:
        class, the input data with an additional column

    """
    np.random.seed(800)  # make sure that it always chooses the same trials as test trials

    for session_id in range(len(all_data)):
        trials_in_session = all_data[session_id]['spks']  # [# of neurons x trials x time (2.5 sec)]
        number_of_trials = trials_in_session.shape[1]  # number of trials in session
        # generate random series of 0s and 1s, make sure to set seed
        trial_types = make_binary_array_with_trial_types(number_of_trials, test_percentage)
        all_data[session_id]['test_data'] = trial_types

    return all_data


def get_test_data(all_data):
    # filter data set and return trials that are in the test data
    for session_id in range(len(all_data)):
        trial_types = all_data[session_id]['test_data']
        indices_of_test_trials = np.where(trial_types == 1)[0]
        all_data[session_id]['spks'] = all_data[session_id]['spks'][:, indices_of_test_trials, :]
        all_data[session_id]['wheel'] = all_data[session_id]['wheel'][:, indices_of_test_trials, :]
        all_data[session_id]['pupil'] = all_data[session_id]['pupil'][:, indices_of_test_trials, :]
        all_data[session_id]['lfp'] = all_data[session_id]['lfp'][:, indices_of_test_trials, :]
        all_data[session_id]['contrast_right'] = all_data[session_id]['contrast_right'][indices_of_test_trials]
        all_data[session_id]['contrast_left'] = all_data[session_id]['contrast_left'][indices_of_test_trials]
        all_data[session_id]['gocue'] = all_data[session_id]['gocue'][indices_of_test_trials]
        all_data[session_id]['response_time'] = all_data[session_id]['response_time'][indices_of_test_trials]
        all_data[session_id]['feedback_time'] = all_data[session_id]['feedback_time'][indices_of_test_trials]
        all_data[session_id]['feedback_type'] = all_data[session_id]['feedback_type'][indices_of_test_trials]
        all_data[session_id]['response'] = all_data[session_id]['response'][indices_of_test_trials]
        all_data[session_id]['test_data'] = all_data[session_id]['test_data'][indices_of_test_trials]

    return all_data


def get_training_data(all_data):
    # filter data set and return trials that are in the training data
    for session_id in range(len(all_data)):
        trial_types = all_data[session_id]['test_data']
        indices_of_training_trials = list(np.where(trial_types == 0)[0])
        all_data[session_id]['spks'] = all_data[session_id]['spks'][:, indices_of_training_trials, :]
        all_data[session_id]['wheel'] = all_data[session_id]['wheel'][:, indices_of_training_trials, :]
        all_data[session_id]['pupil'] = all_data[session_id]['pupil'][:, indices_of_training_trials, :]
        all_data[session_id]['lfp'] = all_data[session_id]['lfp'][:, indices_of_training_trials, :]
        all_data[session_id]['contrast_right'] = all_data[session_id]['contrast_right'][indices_of_training_trials]
        all_data[session_id]['contrast_left'] = all_data[session_id]['contrast_left'][indices_of_training_trials]
        all_data[session_id]['gocue'] = all_data[session_id]['gocue'][indices_of_training_trials]
        all_data[session_id]['response_time'] = all_data[session_id]['response_time'][indices_of_training_trials]
        all_data[session_id]['feedback_time'] = all_data[session_id]['feedback_time'][indices_of_training_trials]
        all_data[session_id]['feedback_type'] = all_data[session_id]['feedback_type'][indices_of_training_trials]
        all_data[session_id]['response'] = all_data[session_id]['response'][indices_of_training_trials]
        all_data[session_id]['test_data'] = all_data[session_id]['test_data'][indices_of_training_trials]
    return all_data


def main():
    all_data = load_data.load_mouse_data()
    all_data_labelled = label_test_and_training_data(all_data)
    test_data = np.array(all_data_labelled, copy=True)
    training_data = np.array(all_data_labelled, copy=True)
    test_data = get_test_data(test_data)
    training_data = get_training_data(training_data)

    print('neuron')


if __name__ == '__main__':
    main()
