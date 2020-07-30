import matplotlib.pylab as plt
import numpy as np
import os
import split_test_and_training_data

dirname = os.path.dirname(__file__)


def add_dummy_choices(session):
    number_of_neurons = session['spks'].shape[0]
    is_choice_neuron = np.random.randint(2, size=number_of_neurons)
    session['is_right_choice_neuron'] = is_choice_neuron  # array of 0s and 1s
    is_choice_neuron = np.random.randint(2, size=number_of_neurons)
    session['is_left_choice_neuron'] = is_choice_neuron  # array of 0s and 1s
    return session


def add_dummy_predictions_for_trials(session):
    number_of_trials = session['spks'].shape[1]
    number_of_neurons = session['spks'].shape[0]
    trial_type = np.random.randint(-1, high=2, size=(number_of_neurons, number_of_trials))
    session['response_prediction_right'] = trial_type
    trial_type = np.random.randint(-1, high=2, size=(number_of_neurons, number_of_trials))
    session['response_prediction_left'] = trial_type

    return session


def check_if_dummy_choices_are_needed_and_add_them(dummy_choice, dummy_prediction, session):
    if dummy_choice:
        session = add_dummy_choices(session)
    if dummy_prediction:
        session = add_dummy_predictions_for_trials(session)
    return session


def plot_proportions_of_correct_predictions(proportions, file_name):
    proportions.sort()
    plt.figure()
    plt.plot(np.arange(len(proportions)), proportions)
    plt.xlabel('Choice neuron id')
    plt.ylabel('Proportion of correctly predicted trials')
    plt.title('Correct predictions per neuron')
    plt.savefig(dirname + '/figures/' + file_name + '.png')
    plt.close()

    plt.figure()
    plt.hist(proportions)
    plt.xlabel('Proportion of correctly predicted trials')
    plt.ylabel('Number of neurons')
    plt.title('Correct predictions per neuron')
    plt.savefig(dirname + '/figures/' + file_name + '_hist.png')
    plt.close()


def get_proportion_of_correct_predictions_for_neuron(train_data, session, type_neuron, neuron_index):
    predictions_of_neuron = train_data[session]['response_prediction_' + type_neuron][neuron_index, :]
    good_prediction = (train_data[session]['response'] == predictions_of_neuron).sum()
    proportion_of_good_predictions = good_prediction / len(predictions_of_neuron)
    return proportion_of_good_predictions


def get_proportion_of_correct_predictions(train_data, type_neuron):
    good_predictions_all_sessions = []
    for session in range(len(train_data)):
        neuron_type = 'is_' + type_neuron + '_choice_neuron'
        is_neuron_type = train_data[session][neuron_type]
        for neuron_index, is_right in enumerate(is_neuron_type):
            if is_right == 1:  # this is right choice neuron
                proportion_of_good_predictions = get_proportion_of_correct_predictions_for_neuron(train_data, session, type_neuron, neuron_index)
                good_predictions_all_sessions.extend([proportion_of_good_predictions])
    plot_proportions_of_correct_predictions(good_predictions_all_sessions,
                                            'proportion_of_good_predictions_' + type_neuron)


def count_choice_neurons_per_region(train_data, type_neuron):
    right_choice_neurons_areas = []
    for session in range(len(train_data)):
        brain_regions = train_data[session]['brain_area']
        neuron_type = 'is_' + type_neuron + '_choice_neuron'
        is_neuron_type = train_data[session][neuron_type] == 1
        brain_regions_choice = brain_regions[is_neuron_type]
        right_choice_neurons_areas.extend(brain_regions_choice)
    plot_number_of_right_choice_neurons_per_region(right_choice_neurons_areas, 'right_choice_neurons_per_area')
    return


def get_proportion_of_correct_predictions_per_region(train_data, type_neuron):
    proportion_of_good_predictions_neuron = []  # number of choice neurons length
    brain_areas_for_each_choice = []  # number of choice neurons length

    for session in range(len(train_data)):
        brain_areas = train_data[session]['brain_area']
        neuron_type = 'is_' + type_neuron + '_choice_neuron'
        is_neuron_type = train_data[session][neuron_type] == 1
        brain_regions_choice = brain_areas[is_neuron_type]
        for neuron_index, is_right in enumerate(is_neuron_type):
            if is_right == 1:  # this is right choice neuron
                proportion_of_good_predictions = get_proportion_of_correct_predictions_for_neuron(train_data, session, type_neuron, neuron_index)

                # this_brain_area = np.where(list_brain_areas == brain_areas[neuron_index])

                proportion_of_good_predictions_neuron.extend([proportion_of_good_predictions])

        brain_areas_for_each_choice.extend(brain_regions_choice)
    # convert to numpy arrays
    proportion_of_good_predictions_neuron = np.array(proportion_of_good_predictions_neuron)
    brain_areas_for_each_choice = np.array(brain_areas_for_each_choice)
    plot_proportion_of_correct_predictions_per_region(proportion_of_good_predictions_neuron, brain_areas_for_each_choice, 'proportion_correct_predictions_per_area')

    return


def plot_proportion_of_correct_predictions_per_region(proportion_of_good_predictions_neuron,
                                                      brain_areas_for_each_choice, file_name):

    proportion_of_good_predictions_per_area = []
    brain_areas = np.unique(brain_areas_for_each_choice)
    for area in brain_areas:
        idx_this_area = np.where(brain_areas_for_each_choice == area)
        this_area_proportion_mean = np.mean(proportion_of_good_predictions_neuron[idx_this_area])
        proportion_of_good_predictions_per_area.append(this_area_proportion_mean)

    plt.figure()
    plt.bar(brain_areas, proportion_of_good_predictions_per_area)
    plt.ylabel('Proportion of correct choices')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(dirname + '/figures/' + file_name + '.png')
    plt.close()

    return


def plot_number_of_right_choice_neurons_per_region(right_choice_neurons_areas, file_name):
    brain_areas, counts_in_area = np.unique(right_choice_neurons_areas, return_counts=True)
    plt.figure()
    plt.bar(brain_areas, counts_in_area)
    plt.xticks(rotation=45)
    plt.ylabel('Number of neurons')
    plt.title('Number of right choice neurons per region')
    plt.tight_layout()
    plt.savefig(dirname + '/figures/' + file_name + '.png')
    plt.close()


def do_descriptive_stats(dummy_choice=False, dummy_prediction=False):
    # load the training data
    train_data, test_data = split_test_and_training_data.train_test_split(train_size=0.8)

    for session in range(len(train_data)):
        train_data[session] = check_if_dummy_choices_are_needed_and_add_them(dummy_choice, dummy_prediction, train_data[session])

    # Plot brain region vs number of choice neuron in each brain region (bar chart)
    count_choice_neurons_per_region(train_data, 'right')
    get_proportion_of_correct_predictions(train_data, 'right')
    get_proportion_of_correct_predictions(train_data, 'left')
    get_proportion_of_correct_predictions_per_region(train_data, 'right')

    # todo: the next thing could be to plot the avg proportion of correct guesses by brain region

    # Look at if motor regions have choice neurons (falls into error correction hypothesis)
    # Pull out which choice neurons encode left v right turn
    # Whether choice neuron predicted the correct turn prediction (compare the identity of choice neuron with the correct
    # wheel movement/ percentage of correct predictions)

    # Whether the mouse executed the same prediction as the choice neuron i.e. left turn choice neuron and then left
    # turn wheel movement
    # proportion of mismatch trials per neuron - done for right choice neurons

    pass


def main():
    do_descriptive_stats(dummy_choice=True, dummy_prediction=True)


if __name__ == '__main__':
    main()
