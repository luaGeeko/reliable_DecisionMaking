""" module to test functions for spliting of dataset """

import pytest
from split_test_and_training_data import create_dataset_split_indices, get_training_data, train_test_split, get_test_data
from load_data import load_mouse_data
import numpy as np

def test_create_dataset_split_indices():
    data = load_mouse_data()
    train_size = 0.7
    session_id = 3
    trials_split_dict = create_dataset_split_indices(data, train_size)
    total_trials_in_sessionid = data[session_id]['spks'].shape[1]
    assert isinstance(trials_split_dict, dict)
    split_found = trials_split_dict[session_id]['train'].shape[0]
    split_percent = int(train_size * total_trials_in_sessionid)
    assert (split_found == split_percent)
    
    #train_data = data[session_id]['spks'][:, train_trials, :]
    #sanity_check = np.equal(data[session_id]['spks'][:, train_trials[0], :], train_data[:, 0, :])
    #print (np.count_nonzero(sanity_check))


def test_get_test_and_training_data():
    data = load_mouse_data()
    train_size = 0.7
    session_id = 6
    training_data, test_data = train_test_split(train_size)
    num_of_trials_all_example = data[session_id]['spks'].shape[1]
    num_of_trials_test_example = test_data[session_id]['spks'].shape[1]
    num_of_trials_training_example = training_data[session_id]['spks'].shape[1]

    assert num_of_trials_all_example == (num_of_trials_test_example + num_of_trials_training_example)

    num_of_trials_all_example = data[session_id]['contrast_right'].shape[0]
    num_of_trials_test_example = test_data[session_id]['contrast_right'].shape[0]
    num_of_trials_training_example = training_data[session_id]['contrast_right'].shape[0]

    assert num_of_trials_all_example == (num_of_trials_test_example + num_of_trials_training_example)


def test_get_training_data():
    # check if it splits the same way every time
    data = load_mouse_data()
    session_id = 2
    train_size = 0.6
    trials_split_dict = create_dataset_split_indices(data, train_size)
    training_data_1 = get_training_data(data, trials_split_dict)
    training_data_2 = get_training_data(data, trials_split_dict)

    assert np.allclose(training_data_1[session_id]['contrast_right'], training_data_2[session_id]['contrast_right'], rtol=1e-05, atol=1e-08, equal_nan=False)


def test_get_test_data():
    # check if it splits the same way every time
    data = load_mouse_data()
    session_id = 10
    train_size = 0.8
    trials_split_dict = create_dataset_split_indices(data, train_size)
    test_data_1 = get_test_data(data, trials_split_dict)
    test_data_2 = get_test_data(data, trials_split_dict)

    assert np.allclose(test_data_1[session_id]['contrast_right'], test_data_2[session_id]['contrast_right'], rtol=1e-05, atol=1e-08, equal_nan=False)


