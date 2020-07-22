import load_data


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
    for session_id in range(len(all_data)):
        trials_in_session = all_data[session_id]['spks']  # [# of neurons x trials x time (2.5 sec)]
        number_of_trials = trials_in_session.shape[-1]  # number of trials in session
        print(number_of_trials)  # sanity check
        # generate random series of 0s and 1s, make sure to set seed
        # add binary array to the data
    return all_data


def get_test_data(all_data):
    # filter data set and return trials that are in the test data
    return all_data


def get_training_data(all_data):
    # filter data set and return trials that are in the training data
    return all_data


def main():
    all_data = load_data.load_mouse_data()
    all_data_labelled = label_test_and_training_data(all_data)


if __name__ == '__main__':
    main()
