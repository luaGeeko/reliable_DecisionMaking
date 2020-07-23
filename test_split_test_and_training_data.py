import split_test_and_training_data
import load_data
import numpy as np


def test_make_binary_array_with_trial_types():
    number_of_trials = 20
    test_percentage = 30
    expected_number_of_1s = 6

    result = split_test_and_training_data.make_binary_array_with_trial_types(number_of_trials, test_percentage)
    assert np.sum(result) == expected_number_of_1s


def test_get_test_and_training_data():
    all_data = load_data.load_mouse_data()
    test_data, training_data = split_test_and_training_data.get_test_and_training_data(all_data)
    num_of_trials_all_example = all_data[6]['spks'].shape[1]
    num_of_trials_test_example = test_data[6]['spks'].shape[1]
    num_of_trials_training_example = training_data[6]['spks'].shape[1]

    assert num_of_trials_all_example == (num_of_trials_test_example + num_of_trials_training_example)

    num_of_trials_all_example = all_data[2]['contrast_right'].shape[0]
    num_of_trials_test_example = test_data[2]['contrast_right'].shape[0]
    num_of_trials_training_example = training_data[2]['contrast_right'].shape[0]

    assert num_of_trials_all_example == (num_of_trials_test_example + num_of_trials_training_example)


def test_get_training_data():
    # check if it splits the same way every time
    all_data = load_data.load_mouse_data()
    training_data_1 = split_test_and_training_data.get_training_data(all_data)
    training_data_2 = split_test_and_training_data.get_training_data(all_data)

    assert np.allclose(training_data_1[2]['contrast_right'], training_data_2[2]['contrast_right'], rtol=1e-05, atol=1e-08, equal_nan=False)


def test_get_test_data():
    # check if it splits the same way every time
    all_data = load_data.load_mouse_data()
    test_data_1 = split_test_and_training_data.get_test_data(all_data)
    test_data_2 = split_test_and_training_data.get_test_data(all_data)

    assert np.allclose(test_data_1[2]['contrast_left'], test_data_2[2]['contrast_left'], rtol=1e-05, atol=1e-08, equal_nan=False)

