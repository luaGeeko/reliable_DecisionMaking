import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


dirname = os.path.dirname(__file__)


def compute_accuracy(X, y, model):
    """Compute accuracy of classifier predictions.

    Args:
      X (2D array): Data matrix
      y (1D array): Label vector
      model (sklearn estimator): Classifier with trained weights.
    Returns:
      accuracy (float): Proportion of correct predictions.
    """
    y_pred = model.predict(X)
    accuracy = (y == y_pred).mean()

    return accuracy


def linear_regression_for_two_choices_for_neuron(spikes, choices):
    """
    choices: two possible behavioural outcomes (could be left vs right or right vs no go etc)
    spikes: spikes on trials for individual neuron
    """

    # First define the model
    log_reg = LogisticRegression(penalty="none")

    # Then fit it to data
    # log_reg.fit(spikes, choices)
    # y_pred = log_reg.predict(spikes)

    # train_accuracy = compute_accuracy(spikes, choices, log_reg)
    # print(f"Accuracy on the training data: {train_accuracy:.2%}")
    accuracies = cross_val_score(LogisticRegression(penalty='l2', solver='saga', max_iter=3000), spikes, choices, cv=8)  # k=8 cross-validation
    # print('Accuracy: ' + str(np.round(accuracies.mean(), 2)))
    return accuracies.mean()


def get_data_from_a_dummy_sesion(session_index=0):
    dataset_folder_path = os.path.join(dirname, "dummy_data")
    # list of the dataset files in the dataset folder
    files = os.listdir(dataset_folder_path)
    # get absolute path of dataset files
    file_paths = [os.path.join(dataset_folder_path, i) for i in files]
    all_data = np.array([])
    for file_id in file_paths:
        all_data = np.hstack((all_data, np.load(file_id, allow_pickle=True)))
    return all_data[session_index]  # one dummy session


def get_data_from_neuron(session, neuron_index):
    return np.nan_to_num(session['spks'][neuron_index, :, :])


def get_indices_for_trial_types(dummy_session):
    responses = dummy_session['response']
    right_mask = responses == -1
    left_mask = responses == 1
    no_go_mask = responses == 0
    right_trials_indices = np.where(right_mask == True)[0]
    left_trials_indices = np.where(left_mask == True)[0]
    no_go_trials_indices = np.where(no_go_mask == True)[0]
    return right_trials_indices, left_trials_indices, no_go_trials_indices


def get_accuracy_predicting_left_vs_right(dummy_session, dummy_neuron):
    # keep trials with left or right choices only
    right_trials_indices, left_trials_indices, no_go_trials_indices = get_indices_for_trial_types(dummy_session)
    # print('check the accuracy when distinguishing left from right:')
    right_and_left_trials_indices = np.append(right_trials_indices, left_trials_indices)
    included_trials_of_dummy_neuron = dummy_neuron[right_and_left_trials_indices, :]  # only left and right trials
    included_decisions = dummy_session['response'][right_and_left_trials_indices]  # list of left and right trials
    accuracy_left_vs_right = linear_regression_for_two_choices_for_neuron(included_trials_of_dummy_neuron,
                                                                          included_decisions)
    return accuracy_left_vs_right


def get_accuracy_predicting_left_vs_no_go(dummy_session, dummy_neuron):
    # keep trials with left and no go choices
    right_trials_indices, left_trials_indices, no_go_trials_indices = get_indices_for_trial_types(dummy_session)
    left_and_nogo_trials_indices = np.append(no_go_trials_indices, left_trials_indices)
    included_trials_of_dummy_neuron = dummy_neuron[left_and_nogo_trials_indices, :]  # only left and right trials
    included_decisions = dummy_session['response'][left_and_nogo_trials_indices]  # list of left and right trials
    accuracy_left_vs_nogo = linear_regression_for_two_choices_for_neuron(included_trials_of_dummy_neuron,
                                                                         included_decisions)
    return accuracy_left_vs_nogo


def get_accuracy_predicting_right_vs_no_go(dummy_session, dummy_neuron):
    # keep trials with right and no go choices
    right_trials_indices, left_trials_indices, no_go_trials_indices = get_indices_for_trial_types(dummy_session)
    right_and_nogo_trials_indices = np.append(no_go_trials_indices, right_trials_indices)
    included_trials_of_dummy_neuron = dummy_neuron[right_and_nogo_trials_indices, :]  # only left and right trials
    included_decisions = dummy_session['response'][right_and_nogo_trials_indices]  # list of left and right trials
    accuracy_right_vs_nogo = linear_regression_for_two_choices_for_neuron(included_trials_of_dummy_neuron,
                                                                         included_decisions)
    return accuracy_right_vs_nogo


def guess_type(left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo):
    if left_vs_right_accuracy > 0.8:
        if accuracy_left_vs_nogo > accuracy_right_vs_nogo:
            neuron_type = 'left_choice'
            print('left')
        else:
            neuron_type = 'right_choice'
            print('right')
    else:
        neuron_type = 'not_choice'
        print('not choice')

    if accuracy_left_vs_nogo > 0.65 and accuracy_right_vs_nogo > 0.65:
        neuron_type = 'action'
        print('not a choice neuron, cannot distinguish left from right')

    return neuron_type


def compare_prediction_accuracies_for_neuron(dummy_session, neuron_index):
    dummy_neuron = get_data_from_neuron(dummy_session, neuron_index)
    print('type of neuron selected for analysis: ' + dummy_session['dummy_type'][neuron_index])
    left_vs_right_accuracy = get_accuracy_predicting_left_vs_right(dummy_session, dummy_neuron)
    # print('check the accuracy when distinguishing left from no go:')
    accuracy_left_vs_nogo = get_accuracy_predicting_left_vs_no_go(dummy_session, dummy_neuron)
    # print('check the accuracy when distinguishing right from no go:')
    accuracy_right_vs_nogo = get_accuracy_predicting_right_vs_no_go(dummy_session, dummy_neuron)
    return np.round(left_vs_right_accuracy, 2), np.round(accuracy_left_vs_nogo, 2), np.round(accuracy_right_vs_nogo, 2)


def compare_predicted_to_ground_truth_type(ground_truth, predicted):
    correct_guesses = 0
    keys, counts = np.unique(predicted, return_counts=True)
    choice_index = np.where(keys == 'choice')
    number_of_times_it_guessed_choice = counts[choice_index]
    for index, neuron_type in enumerate(ground_truth):
        if neuron_type == predicted[index]:
            correct_guesses += 1
        elif predicted[index] == 'not_choice':
            if neuron_type == 'random':
                correct_guesses += 1
            if neuron_type == 'peak_at_response':
                correct_guesses += 1
            if neuron_type == 'ramp_to_action':
                correct_guesses += 1
            if neuron_type == 'left_contrast_higher':
                correct_guesses += 1
            if neuron_type == 'right_contrast_higher':
                correct_guesses += 1

    prediction_accuracy = correct_guesses / len(ground_truth)
    print('prediction accuracy is:' + str(np.round(prediction_accuracy, 2)))
    print('model guessed choice but not side: ' + str(np.round(number_of_times_it_guessed_choice)))


def identify_neuron_types():
    # get trials for a single dummy neuron
    dummy_session = get_data_from_a_dummy_sesion()
    # this is the first neuron from the dummy session:
    print('Analyze two example neurons first.')
    neuron_index = 0  # we will try to decide whether this is a choice neuron
    left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo = compare_prediction_accuracies_for_neuron(dummy_session, neuron_index)
    print('accuracies (left vs right, left vs no go, right vs no go):' + str(left_vs_right_accuracy) + ' ' + str(accuracy_left_vs_nogo) + ' ' + str(accuracy_right_vs_nogo))
    neuron_type = guess_type(left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo)
    print('----------------------------------------------------------------------')
    neuron_index = 55  # we will try to decide whether this is a choice neuron
    left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo = compare_prediction_accuracies_for_neuron(dummy_session, neuron_index)
    print('accuracies (left vs right, left vs no go, right vs no go):' + str(left_vs_right_accuracy) + ' ' + str(accuracy_left_vs_nogo) + ' ' + str(accuracy_right_vs_nogo))
    neuron_type = guess_type(left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo)

    print('Analyze all neurons from this session and check accuracy of predictions:')
    number_of_neurons = dummy_session['spks'].shape[0]
    neuron_type_guesses = []
    for neuron_index in range(number_of_neurons):
        left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo = compare_prediction_accuracies_for_neuron(dummy_session, neuron_index)
        neuron_type = guess_type(left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo)
        neuron_type_guesses.append(neuron_type)

    compare_predicted_to_ground_truth_type(dummy_session['dummy_type'], neuron_type_guesses)


def main():
    identify_neuron_types()


if __name__ == '__main__':
    main()


