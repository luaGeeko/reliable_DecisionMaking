""" This module splits the dataset into training, validation and test sets. Test set is never to be seen by the model. While training, validation set is used to validate 
    how well model performs. """

import copy
from load_data import load_mouse_data
import numpy as np
import coloredlogs, logging

# debugging
logger = logging.getLogger("split_dataset")
coloredlogs.install(level='INFO', logger=logger)

# random seed for reproducibility of experiements with dataset splits
np.random.seed(75)

def create_dataset_split_indices(data, train_size):
    """ 
    splits the data with given percentage in two parts [ train / test]. the given percentage is kept for training set 
    and remaining for test set
    Args:
        train_size (float) : train percentage to be kept for training data
    Returns:
        dict holding the session ids as keys and their corresponding trial indices created for train and test set 
    """
    dataset_split_indices_holder = {}
    # NOTE: in matlab the index starts from 1 in python its starts from 0
    for session_id in range(data.shape[0]):
        num_spike_trials = data[session_id]['spks']
        # get trails for the neurons
        session_id_trial_indices = np.random.permutation(num_spike_trials.shape[1])
        # NOTE: split done for only train and test sets, I think validation set is also useful can easily be modified later
        #training_idx, temp_valid_idx = np.split(session_id_indices, [int(train_per * session_id_indices.shape[0])])
        training_trials_idx, test_trails_idx = np.split(session_id_trial_indices, [int(train_size * session_id_trial_indices.shape[0])])
        assert session_id_trial_indices.shape[0] == (training_trials_idx.shape[0] + test_trails_idx.shape[0])
        dataset_split_indices_holder[session_id] = {'train': training_trials_idx, 'test': test_trails_idx}

    return dataset_split_indices_holder


def get_data_for_trial_indices(data, session_id, trial_indices):
    """ 
    extracts the indices from data according to given set type [train/test]
    Args:
        data (numpy array) : simulated or mouse data
        session_id (int) : session id for which data is to reformed
        split_indices (dict) : a dictionary holding indices for given set type [train/test] for all the sessions
    Returns:
        numpy array holding data for either training / test set for all sessions
    """
    temp_holder = {}
    temp_holder['spks'] = data[session_id]['spks'][:, trial_indices, :]
    temp_holder['wheel'] = data[session_id]['wheel'][:, trial_indices, :]
    temp_holder['pupil'] = data[session_id]['pupil'][:, trial_indices, :]
    temp_holder['lfp'] = data[session_id]['lfp'][:, trial_indices, :]
    temp_holder['contrast_right'] = data[session_id]['contrast_right'][trial_indices]
    temp_holder['contrast_left'] = data[session_id]['contrast_left'][trial_indices]
    temp_holder['gocue'] = data[session_id]['gocue'][trial_indices]
    temp_holder['response_time'] = data[session_id]['response_time'][trial_indices]
    temp_holder['feedback_time'] = data[session_id]['feedback_time'][trial_indices]
    temp_holder['feedback_type'] = data[session_id]['feedback_type'][trial_indices]
    temp_holder['response'] = data[session_id]['response'][trial_indices]
    return temp_holder

def get_training_data(data, split_indices):
    """ creates training set """
    train_data = {}
    for session_id in range(data.shape[0]):
        train_data[session_id] = get_data_for_trial_indices(data, session_id, split_indices[session_id]['train'])
    logger.info("training data created")
    return train_data

def get_test_data(data, split_indices):
    """ creates test set """
    test_data = {}
    for session_id in range(data.shape[0]):
        test_data[session_id] = get_data_for_trial_indices(data, session_id, split_indices[session_id]['test'])
    logger.info("test data created")
    return test_data


def train_test_split(train_size: float = 0.8):
    """ split the dataset into training and test set """
    # load dataset
    data = load_mouse_data()
    split_trial_indices = create_dataset_split_indices(data, train_size)
    train_data = get_training_data(data, split_trial_indices)
    test_data = get_test_data(data, split_trial_indices)
    return train_data, test_data