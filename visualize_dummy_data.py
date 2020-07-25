import numpy as np
import os
import visualize

dirname = os.path.dirname(__file__)


def visualize_dummy_data():
    dataset_folder_path = os.path.join(dirname, "dummy_data")
    # list of the dataset files in the dataset folder
    files = os.listdir(dataset_folder_path)[:2]  # get first few only
    # get absolute path of dataset files
    file_paths = [os.path.join(dataset_folder_path, i) for i in files]
    all_data = np.array([])
    for file_id in file_paths:
        all_data = np.hstack((all_data, np.load(file_id, allow_pickle=True)['dat']))
    return all_data
    visualize.plot_population_average(example_session, file_name='population_average_dummy4')


def main():
    visualize_dummy_data()


if __name__ == '__main__':
    main()
