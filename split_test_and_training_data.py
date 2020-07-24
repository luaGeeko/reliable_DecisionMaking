import copy
import load_data
import numpy as np


def make_binary_array_with_trial_types(number_of_trials, test_percentage):
    """
        Makes binary array where the length of the array is the number of trials and the proportion of 1s is
        test percentage.

        Args:
            number_of_trials (int) : number of trials
            test_percentage (float) : the percentage of trials per neuron that will be test trials
                                       0 >= test_percentage <= 1
        Returns:
            binary list where 1 represents a test trial and 0 a training trial

        """
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
    print('Labelling data for splitting.')
    print('WARNING: the passive trials will not be split by this.')
    np.random.seed(800)  # make sure that it always chooses the same trials as test trials

    for session_id in range(len(all_data)):
        trials_in_session = all_data[session_id]['spks']  # [# of neurons x trials x time (2.5 sec)]
        number_of_trials = trials_in_session.shape[1]  # number of trials in session
        # generate random series of 0s and 1s
        trial_types = make_binary_array_with_trial_types(number_of_trials, test_percentage)
        all_data[session_id]['test_data'] = trial_types

    return all_data


def get_data_for_trial_indices(all_data, session_id, trial_indices):
    """
    Overwrites data with data from selected trials only

    Args:
        all_data (class) : all simulated or mouse data
        session_id (int) : id of session to update
        trial_indices (list) : list of trial indices to keep in data

    Returns:
        numpy array, session with updated data
    """
    all_data[session_id]['spks'] = all_data[session_id]['spks'][:, trial_indices, :]
    all_data[session_id]['wheel'] = all_data[session_id]['wheel'][:, trial_indices, :]
    all_data[session_id]['pupil'] = all_data[session_id]['pupil'][:, trial_indices, :]
    all_data[session_id]['lfp'] = all_data[session_id]['lfp'][:, trial_indices, :]
    all_data[session_id]['contrast_right'] = all_data[session_id]['contrast_right'][trial_indices]
    all_data[session_id]['contrast_left'] = all_data[session_id]['contrast_left'][trial_indices]
    all_data[session_id]['gocue'] = all_data[session_id]['gocue'][trial_indices]
    all_data[session_id]['response_time'] = all_data[session_id]['response_time'][trial_indices]
    all_data[session_id]['feedback_time'] = all_data[session_id]['feedback_time'][trial_indices]
    all_data[session_id]['feedback_type'] = all_data[session_id]['feedback_type'][trial_indices]
    all_data[session_id]['response'] = all_data[session_id]['response'][trial_indices]
    all_data[session_id]['test_data'] = all_data[session_id]['test_data'][trial_indices]
    return all_data[session_id]


def get_test_data(all_data_in):
    """
    Get test data set. The seed is set so this should always give the same result

    Args:
        all_data (class) : all simulated or mouse data

    Returns:
        test data in the same format as the input data
    """

    all_data = copy.deepcopy(all_data_in)
    all_data = label_test_and_training_data(all_data)
    # filter data set and return trials that are in the test data
    for session_id in range(len(all_data)):
        trial_types = all_data[session_id]['test_data']
        indices_of_test_trials = np.where(trial_types == 1)[0]
        all_data[session_id] = get_data_for_trial_indices(all_data, session_id, indices_of_test_trials)
    return all_data


def get_training_data(all_data_in):
    """
    Get training data set. The seed is set so this should always give the same result

    Args:
        all_data (class) : all simulated or mouse data

    Returns:
        training data in the same format as the input data
    """

    all_data = copy.deepcopy(all_data_in)
    all_data = label_test_and_training_data(all_data)
    # filter data set and return trials that are in the training data
    for session_id in range(len(all_data)):
        trial_types = all_data[session_id]['test_data']
        indices_of_training_trials = list(np.where(trial_types == 0)[0])
        all_data[session_id] = get_data_for_trial_indices(all_data, session_id, indices_of_training_trials)
    return all_data


def get_test_and_training_data(all_data):
    """
    Return test and training data sets.
    Args:
        all_data (class, numpy array of dictionaries) : all simulated or mouse data

    Returns:
        training and test data in the same format as the input data
    """
    test_data = get_test_data(all_data)
    training_data = get_training_data(all_data)
    return test_data, training_data


# this is just here for testing
def main():
    all_data = load_data.load_mouse_data()
    test_data = get_test_data(all_data)
    training_data = get_training_data(all_data)


if __name__ == '__main__':
    main()
