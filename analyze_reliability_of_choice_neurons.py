# import descriptive_stats
import generate_synthetic_data
import load_data
# import pre_process_data
# import modules both models
import split_test_and_training_data


def main():
    # generate synthetic data
    generate_synthetic_data.generate_dummy_data()  # saves the simulated data
    # load data
    mouse_data = load_data.load_mouse_data()
    # split data
    train_data, test_data = split_test_and_training_data.train_test_split(train_size=0.8)
    # evaluate model #1 on dummy data
    ### this is the linear regression model on the level of individual neurons

    # do dimensionality reduction on real and dummy data
    # reduced_dim_training_dummy = pre_process_data.dimensionality_reduction(train_data, 0.9)
    # reduced_dim_training = pre_process_data.dimensionality_reduction(train_data, 0.9)
    # reduced_dim_test_dummy = pre_process_data.dimensionality_reduction(test_data, 0.9)
    # reduced_dim_test = pre_process_data.dimensionality_reduction(test_data, 0.9)
    # evaluate model on dummy data
    # the model should take the training and test data sets as an input and return them with the new columns added
    ### this is the GLM, it should add the extra columns the descriptive stats needs

    # evaluate model #1 on observed data
    # evaluate model #2 on observed data

    # plot results for all
    # input should be the data / dummy data with the result columns
    # descriptive_stats.descriptive_stats.do_descriptive_stats()


if __name__ == '__main__':
    main()