import descriptive_stats
import generate_synthetic_data
import load_data
import pre_process_data
import identify_choice_neurons
import split_test_and_training_data


def analyze_data(make_dummy=True, run_lin_reg_model=True):
    if make_dummy:
        # generate synthetic data
        generate_synthetic_data.generate_dummy_data()  # saves the simulated data

    # evaluate model #1 on dummy data
    # this is the linear regression model on the level of individual neurons. it will save the results
    if run_lin_reg_model:
        identify_choice_neurons.identify_neuron_types()  # todo update once Shruti is finished

    # plot results for choice neuron reliability
    descriptive_stats.do_descriptive_stats()


def main():
    analyze_data(make_dummy=False, run_lin_reg_model=False)


if __name__ == '__main__':
    main()