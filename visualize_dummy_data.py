import numpy as np
import os
import visualize

dirname = os.path.dirname(__file__)


def visualize_dummy_data():
    dataset_folder_path = os.path.join(dirname, "dummy_data")
    # list of the dataset files in the dataset folder
    files = os.listdir(dataset_folder_path)  # get first few only
    # get absolute path of dataset files
    file_paths = [os.path.join(dataset_folder_path, i) for i in files]
    all_data = np.array([])
    for file_id in file_paths:
        all_data = np.hstack((all_data, np.load(file_id, allow_pickle=True)))
    #visualize.plot_population_average(all_data[0], file_name='population_average_dummy0')
    #visualize.plot_random_examples(all_data[0], file_name='random_examples_dummy0')
    visualize.plot_example_from_each_neuron_type(all_data[0], file_name='type_examples_dummy0')
    visualize.plot_example_from_each_neuron_type_contrast(all_data[0], file_name='contrast_type_examples_dummy0')
    visualize.plot_proportion_of_dummy_types(all_data[0], file_name='proportion_of_dummies0')


def main():
    visualize_dummy_data()


if __name__ == '__main__':
    main()
