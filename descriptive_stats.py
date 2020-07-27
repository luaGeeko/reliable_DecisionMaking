import matplotlib.pylab as plt
import numpy as np
import os
import split_test_and_training_data


dirname = os.path.dirname(__file__)


def add_dummy_choices(session):
    number_of_neurons = session['spks'].shape[0]
    is_choice_neuron = np.random.randint(2, size=number_of_neurons)
    session['is_right_choice_neuron'] = is_choice_neuron  # array of 0s and 1s
    return session


def add_dummy_predictions_for_trials(session):
    number_of_trials = session['spks'].shape[1]
    number_of_neurons = session['spks'].shape[0]
    trial_type = np.random.randint(-1, high=2, size=(number_of_neurons, number_of_trials))
    session['response_prediction'] = trial_type  # array of 0s and 1s
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


def do_descriptive_stats(dummy_choice=False, dummy_prediction=False):
    # load the training data

    train_data, test_data = split_test_and_training_data.train_test_split(train_size=0.8)
    good_predictions_all_sessions = []
    for session in range(len(train_data)):
        train_data[session] = add_dummy_choices(train_data[session])
        train_data[session] = add_dummy_predictions_for_trials(train_data[session])
        print('')
        is_right_choice_neuron = train_data[session]['is_right_choice_neuron']
        right_choice_behaviour = train_data[session]['response'] == -1
        for neuron_index, is_right in enumerate(is_right_choice_neuron):
            if is_right == 1:   # this is right choice neuron
                predictions_of_neuron = train_data[session]['response_prediction'][neuron_index, :]
                good_prediction = (train_data[session]['response'] == predictions_of_neuron).sum()
                proportion_of_good_predictions = good_prediction / len(predictions_of_neuron)
                good_predictions_all_sessions.extend([proportion_of_good_predictions])

    plot_proportions_of_correct_predictions(good_predictions_all_sessions, 'ptoportion_of_good_predictions')

    # todo: the next thing could be to plot the avg proportion of correct guesses by brain region
    # Whether choice neuron predicted the correct turn prediction
    # Plot brain region vs number of choice neuron in each brain region (bar chart)
    # Look at if motor regions have choice neurons (falls into error correction hypothesis)
    # Pull out which choice neurons encode left v right turn
    # Whether choice neuron predicted the correct turn prediction (compare the identity of choice neuron with the correct
    # wheel movement/ percentage of correct predictions)

    # Whether the mouse executed the same prediction as the choice neuron i.e. left turn choice neuron and then left
    # turn wheel movement
    # proportion of mismatch trials per neuron

    pass


def main():
    do_descriptive_stats(dummy_choice=True, dummy_prediction=True)


if __name__ == '__main__':
    main()