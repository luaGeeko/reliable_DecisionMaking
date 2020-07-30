import numpy as np
import matplotlib.pyplot as plt
import os
from load_data import load_mouse_data
#import coloredlogs, logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from split_test_and_training_data import train_test_split
#from sklearn.metrics import classification_report, plot_confusion_matrix, f1_score
import seaborn as sns
from sklearn.metrics import f1_score
import pandas as pd
sns.set(style="white")

dirname = os.path.dirname(__file__)

""" 
CASES : considering for single neuron
1. when left trials mean accuracy > right trials mean accuracy, look into confusion matrix how many times it gets confused by the other trials why does that happen is some cases?
2. when left trials mean accuracy == right trials mean accuracy, means the model was perfectly differentiated between both with no confusion does that mean spikes for each left
and right trials were very differented to be understood
3. when right trials mean accuracy > left trials mean accuracy, there were some cases one of trials got confused with other why this happened means something was not 
differentiable enough between spikes from left trails and right trials
"""

np.random.seed(89)
# debugging
# logger = logging.getLogger("model")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# coloredlogs.install(level='INFO', logger=logger)

classes = ['nogo_choice_neuron', 'left_choice_neuron', 'right_choice_neuron']
classes_to_idx = {classes[i]: i for i in range(len(classes))}
idx_to_classes = {i : classes[i] for i in range(len(classes))}
# logger.info("Class cateogries: {}".format(classes_to_idx))

def prepare_labels_in_correct_format(neuron_data_labels):
    # the labels for trials is [1, 0, -1], for classifying -1 is no sense, we can approach this in categorical way like multiclass classification
    # correcting the label for right choice neuron from -1 to label created above in classes_to_idx
    neuron_data_labels[neuron_data_labels == -1] = classes_to_idx['right_choice_neuron']
    return neuron_data_labels

#NOTE should we check accross all sessions or neurons in each session. also minor imbalance trials is kinda ok for now but big proportion should not be there
#FIXME add assert test for correct format
def check_class_proportion(data, across_sessions: bool = False, single_neuron: bool = False, across_neurons_in_single_session: bool = True):
    """ data is the dictionary with all sessions """
    if across_neurons_in_single_session:
        # take all the trials count left_trials, right_trials, nogo_trials
        random_session_id = np.random.choice(data.shape[0], size=1)[0]
        # logger.info("random session id selected {} ".format(random_session_id))
        session_data = data[random_session_id]
        spike_session_data = session_data['spks']
        assert len(spike_session_data.shape) == 3, "the shape for neurons x trials x timebins is needed but found " + str(spike_session_data.shape)
        right_trials_indices, left_trials_indices, no_go_trials_indices = get_indices_for_trial_types(session_data)
        value_count = [no_go_trials_indices.shape[0], right_trials_indices.shape[0], left_trials_indices.shape[0]]
        plt.bar(classes, value_count, alpha=0.7)
        plt.show()

    elif single_neuron:
        pass

def get_indices_for_trial_types(dummy_session):
    responses = dummy_session['response']
    right_mask = responses == -1
    left_mask = responses == 1
    no_go_mask = responses == 0
    right_trials_indices = np.where(right_mask == True)[0]
    left_trials_indices = np.where(left_mask == True)[0]
    no_go_trials_indices = np.where(no_go_mask == True)[0]
    return right_trials_indices, left_trials_indices, no_go_trials_indices

def get_spike_count(neuron_trials_data):
    spikes_count_all_trials = []
    for i in range(neuron_trials_data.shape[0]):
        # each trial on this neuron
        trial = neuron_trials_data[i]
        spike_count = np.count_nonzero(trial)
        spikes_count_all_trials.append(spike_count)


def logistic_model_V2(train_spikes, train_choices, title: str = None):
    """
    train_choices: ground truth - two possible behavioural outcomes (could be left vs right or right vs no go etc)
    train_spikes: spikes on trials for individual neuron for training 
    test_spikes: test model accuracy based on this test neuron
    """
    # First define the model
    model = LogisticRegression(solver='liblinear', random_state=400, class_weight='balanced').fit(train_spikes, train_choices)
    model_train_score = model.score(train_spikes, train_choices)
    # predictions = model.predict(train_spikes)
    predict_probs = model.predict_proba(train_spikes)
    probabilities_1 = predict_probs[:, 0]
    probabilities_2 = predict_probs[:, 1]
    mean_probs_right, mean_probs_left = predict_probs[:, 0].mean(), predict_probs[:, 1].mean()
    print ("train score {}".format(model_train_score))
    print ("model classes", model.classes_)
    print ("model slope", model.coef_.shape)
    print ("model intercept", model.intercept_)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    x_min, x_max = ax1.get_xlim()
    #ax1.plot([0, x_max], [model.intercept_[0], x_max*model.coef_.reshape(-1).mean()+model.intercept_[0]])
    #plt.show()
    # model evaulation accuracies with differeent splits
    #show_metrics(model, train_spikes, train_choices, predictions, title=title)
    return probabilities_1, probabilities_2


def logistic_model(train_spikes, train_choices, title: str = None):
    """
    train_choices: ground truth - two possible behavioural outcomes (could be left vs right or right vs no go etc)
    train_spikes: spikes on trials for individual neuron for training 
    test_spikes: test model accuracy based on this test neuron
    """
    # First define the model
    model = LogisticRegression(solver='liblinear', random_state=400, class_weight='balanced').fit(train_spikes, train_choices)
    model_train_score = model.score(train_spikes, train_choices)
    predictions = model.predict(train_spikes)

    f_score = f1_score(train_choices, predictions, average='micro')
    #print ("f_score calculated", f_score)
    cross_val_accuracy_scores = cross_val_score(model, train_spikes, train_choices, cv=4)  # k=4 cross-validation
    cross_val_mean, cross_val_std = cross_val_accuracy_scores.mean(), cross_val_accuracy_scores.std()
    #show_metrics(model, train_spikes, train_choices, predictions, title=title)
    return cross_val_mean, f_score


def logistic_model_Firing_rate(train_spikes, train_choices):
    """
    train_choices: ground truth - two possible behavioural outcomes (could be left vs right or right vs no go etc)
    train_spikes: spikes on trials for individual neuron for training 
    test_spikes: test model accuracy based on this test neuron
    """
    # First define the model
    model = LogisticRegression(solver='liblinear', random_state=400, class_weight='balanced').fit(train_spikes, train_choices)
    model_train_score = model.score(train_spikes, train_choices)
    predictions = model.predict(train_spikes)
    # f_score = f1_score(train_choices, predictions, average='micro')
    pred1 = train_spikes[predictions == model.classes_[0], :]
    pred2 = train_spikes[predictions == model.classes_[1], :]
    return pred1, pred2, model.classes_


def get_data_from_a_dummy_sesion(session_index: int = 0):
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


def get_real_data_from_session(session_id: int = 10):
    data = load_mouse_data()
    session_data = data[session_id]
    return session_id


def get_accuracy_predicting_left_vs_right(dummy_session, dummy_train_neuron):
    # keep trials with left or right choices only
    right_trials_indices, left_trials_indices, no_go_trials_indices = get_indices_for_trial_types(dummy_session)
    # print('check the accuracy when distinguishing left from right:')
    right_and_left_trials_indices = np.append(right_trials_indices, left_trials_indices)
    included_trials_of_dummy_neuron = dummy_train_neuron[right_and_left_trials_indices, :]  # only left and right trials
    included_decisions = dummy_session['response'][right_and_left_trials_indices]  # list of left and right trials
    # correct the labels of the classes
    included_decisions[included_decisions == -1] = 2
    model_acc, f_score = logistic_model(included_trials_of_dummy_neuron, included_decisions, 'left_vs_right')
    probs_right, probs_left = logistic_model_V2(included_trials_of_dummy_neuron, included_decisions)
    return model_acc, f_score, probs_right, probs_left


def get_accuracy_predicting_left_vs_no_go(dummy_session, dummy_neuron):
    # keep trials with left and no go choices
    right_trials_indices, left_trials_indices, no_go_trials_indices = get_indices_for_trial_types(dummy_session)
    left_and_nogo_trials_indices = np.append(no_go_trials_indices, left_trials_indices)
    included_trials_of_dummy_neuron = dummy_neuron[left_and_nogo_trials_indices, :]  # only left and right trials
    included_decisions = dummy_session['response'][left_and_nogo_trials_indices]  # list of left and right trials
    model_acc, f_score = logistic_model(included_trials_of_dummy_neuron, included_decisions, 'left_vs_nogo')
    probs_left, probs_no_go = logistic_model_V2(included_trials_of_dummy_neuron, included_decisions)
    return model_acc, f_score, probs_left, probs_no_go


def get_accuracy_predicting_right_vs_no_go(dummy_session, dummy_neuron):
    """ correct the labels """
    # keep trials with right and no go choices
    right_trials_indices, left_trials_indices, no_go_trials_indices = get_indices_for_trial_types(dummy_session)
    right_and_nogo_trials_indices = np.append(no_go_trials_indices, right_trials_indices)
    included_trials_of_dummy_neuron = dummy_neuron[right_and_nogo_trials_indices, :]  # only left and right trials
    included_decisions = dummy_session['response'][right_and_nogo_trials_indices]  # list of left and right trials
     # correct the labels of the classes either 0 or 1
    included_decisions[included_decisions == -1] = 2
    model_acc, f_score = logistic_model(included_trials_of_dummy_neuron, included_decisions, 'right_vs_nogo')
    probs_right, probs_no_go = logistic_model_V2(included_trials_of_dummy_neuron, included_decisions)
    return model_acc, f_score, probs_right, probs_no_go


# evaluating model on f scores basis
def guess_type(left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo):
    if left_vs_right_accuracy > 0.85:
        if accuracy_left_vs_nogo > accuracy_right_vs_nogo:
            neuron_type = 'left_choice'
            print('left')
        elif 0.50 < accuracy_right_vs_nogo <= 0.60 and 0.50 < accuracy_left_vs_nogo <= 0.60:
            neuron_type = 'action'
            print ("can not differentiate between left or right")
        else:
            neuron_type = 'right_choice'
            print('right')
    else:
        neuron_type = 'not_choice'
        print('not choice')

    return neuron_type


def get_predicted_choices_for_trials(probabilities, session):
    predictions = np.zeros(session['spks'].shape[1])
    prediction_probabilities = np.zeros(session['spks'].shape[1])
    right_trials_indices, left_trials_indices, no_go_trials_indices = get_indices_for_trial_types(session)

    r_ng_ind = np.append(right_trials_indices, no_go_trials_indices)
    l_ng_ind = np.append(left_trials_indices, no_go_trials_indices)
    l_r_ind = np.append(left_trials_indices, right_trials_indices)

    no_go_r = np.zeros(session['spks'].shape[1])
    no_go_r[r_ng_ind] = probabilities['probs_no_go_r'][0]

    no_go_l = np.zeros(session['spks'].shape[1])
    no_go_l[l_ng_ind] = probabilities['probs_no_go_l'][0]

    left_r = np.zeros(session['spks'].shape[1])
    left_r[l_r_ind] = probabilities['probs_left_r'][0]

    left_n = np.zeros(session['spks'].shape[1])
    left_n[l_ng_ind] = probabilities['probs_left_n'][0]

    right_l = np.zeros(session['spks'].shape[1])
    right_l[l_r_ind] = probabilities['probs_right_l'][0]

    right_n = np.zeros(session['spks'].shape[1])
    right_n[r_ng_ind] = probabilities['probs_right_n'][0]

    probs_df = pd.DataFrame()
    probs_df['no_go_r'] = no_go_r
    probs_df['no_go_l'] = no_go_l
    probs_df['left_r'] = left_r
    probs_df['left_n'] = left_n
    probs_df['right_r'] = right_l
    probs_df['right_n'] = right_n

    highest_probs = list(probs_df.idxmax(axis=1))
    predictions = []
    for p in highest_probs:
        if "right" in p:
            predictions.append(-1)
        elif "left" in p:
            predictions.append(1)
        elif "no_go" in p:
            predictions.append(0)
    return predictions


def compare_prediction_accuracies_for_neuron(dummy_session, neuron_index):
    dummy_neuron = get_data_from_neuron(dummy_session, neuron_index)
    #print('type of neuron selected for analysis: ' + dummy_session['dummy_type'][neuron_index])
    model0_acc, model0_f_score, probs_right_l, probs_left_r = get_accuracy_predicting_left_vs_right(dummy_session, dummy_neuron)
    # print('check the accuracy when distinguishing left from no go:')
    model1_acc, model1_f_score, probs_left_n, probs_no_go_l = get_accuracy_predicting_left_vs_no_go(dummy_session, dummy_neuron)
    # print('check the accuracy when distinguishing right from no go:')
    model2_acc, model2_f_score, probs_right_n, probs_no_go_r = get_accuracy_predicting_right_vs_no_go(dummy_session, dummy_neuron)
    probabilities_df = pd.DataFrame()
    probabilities_df['probs_right_l'] = [probs_right_l]
    probabilities_df['probs_left_r'] = [probs_left_r]
    probabilities_df['probs_left_n'] = [probs_left_n]
    probabilities_df['probs_right_n'] = [probs_right_n]
    probabilities_df['probs_no_go_l'] = [probs_no_go_l]
    probabilities_df['probs_no_go_r'] = [probs_no_go_r]
    predictions_trials = get_predicted_choices_for_trials(probabilities_df, dummy_session)
    return np.round(model0_f_score, 2), np.round(model1_f_score, 2), np.round(model2_f_score, 2), predictions_trials


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


def run_model_on_real_data_session(session_data):
    # session_data = get_data_from_a_dummy_sesion(session_id)
    print('Analyze all neurons from this session and check accuracy of predictions:')
    number_of_neurons = session_data['spks'].shape[0]
    number_of_trials = session_data['spks'].shape[1]
    neuron_type_guesses = []
    is_right_choice_neuron = []
    is_left_choice_neuron = []
    predictions = np.zeros((number_of_neurons, number_of_trials))
    for neuron_index in range(number_of_neurons):
        left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo, predictions_trials = compare_prediction_accuracies_for_neuron(session_data, neuron_index)
        predictions[neuron_index, :] = predictions_trials
        neuron_type = guess_type(left_vs_right_accuracy, accuracy_left_vs_nogo, accuracy_right_vs_nogo)
        print("neuron_idx: {} neuron type {}".format(neuron_index, neuron_type))
        if neuron_type == 'left':
            is_right_choice_neuron.extend([0])
            is_left_choice_neuron.extend([1])
        elif neuron_type == 'right':
            is_right_choice_neuron.extend([1])
            is_left_choice_neuron.extend([0])
        else:
            is_right_choice_neuron.extend([0])
            is_left_choice_neuron.extend([0])
        neuron_type_guesses.append(neuron_type)
    session_data['is_right_choice_neuron'] = is_right_choice_neuron
    session_data['is_left_choice_neuron'] = is_right_choice_neuron
    session_data['response_prediction_left'] = predictions
    session_data['response_prediction_right'] = predictions
    return session_data
        
    # compare_predicted_to_ground_truth_type(session_data['dummy_type'], neuron_type_guesses)

    
# just tested with this function with dummy data it saves the results in the dictionary format
# example: {session_id : {neuron_id: {"is_right_choice_neuron": 0, "is_left_choice_neuron": 1}}}
def main():
    data = load_mouse_data()
    for session in range(len(data)):
        data[session] = run_model_on_real_data_session(data[session])
        print('')


if __name__ == '__main__':
    main()
