""" module to test for generation of synthetic dataset """

import pytest
from split_test_and_training_data import create_dataset_split_indices, get_training_data, train_test_split, get_test_data
from load_data import load_mouse_data
import numpy as np