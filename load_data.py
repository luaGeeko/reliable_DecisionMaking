import numpy as np
import coloredlogs, logging 
import os

dirname = os.path.dirname(__file__)

# debugging
logger = logging.getLogger("load_data")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
coloredlogs.install(level='INFO', logger=logger)

def load_mouse_data():
    """
    Loads mouse data. The 3 data files should be in the same folder as this script.

    Returns: class, data from all mice
    contains 39 sessions from 10 mice, data from Steinmetz et al, 2019.
    Time bins for all measurements are 10ms, starting 500ms before stimulus onset.
    The mouse had to determine which side has the highest contrast.
    For each dat = alldat[k], you have the following fields:
    dat['mouse_name']: mouse name
    dat['date_exp']: when a session was performed
    dat['spks']: neurons by trials by time bins.
    dat['brain_area']: brain area for each neuron recorded.
    dat['contrast_right']: contrast level for the right stimulus, which is always contralateral to the recorded brain areas.
    dat['contrast_left']: contrast level for left stimulus.
    dat['gocue']: when the go cue sound was played.
    dat['response_times']: when the response was registered, which has to be after the go cue. The mouse can turn the
    wheel before the go cue (and nearly always does!), but the stimulus on the screen won't move before the go cue.
    dat['response']: which side the response was (-1, 0, 1). When the right-side stimulus had higher contrast, the
    correct choice was -1. 0 is a no go response.
    dat['feedback_time']: when feedback was provided.
    dat['feedback_type']: if the feedback was positive (+1, reward) or negative (-1, white noise burst).
    dat['wheel']: exact position of the wheel that the mice uses to make a response, binned at 10ms.
    dat['pupil']: pupil area (noisy, because pupil is very small) + pupil horizontal and vertical position.
    dat['lfp']: recording of the local field potential in each brain area from this experiment, binned at 10ms.
    dat['brain_area_lfp']: brain area names for the LFP channels.
    dat['trough_to_peak']: measures the width of the action potential waveform for each neuron. Widths <=10 samples are
     "putative fast spiking neurons".
    dat['waveform_w']: temporal components of spike waveforms. w@u reconstructs the time by channels action potential
     shape.
    dat['waveform_u]: spatial components of spike waveforms.
    dat['%X%_passive']: same as above for X = {spks, lfp, pupil, wheel, contrast_left, contrast_right} but for passive
    trials at the end of the recording when the mouse was no longer engaged and stopped making responses.
    """
    dataset_folder_path = os.path.join(dirname, "dataset")
    # list of the dataset files in the dataset folder 
    files = os.listdir(dataset_folder_path)
    # get absolute path of dataset files 
    file_paths = [os.path.join(dataset_folder_path, i) for i in files]
    all_data = np.array([])
    for file_id in file_paths:
        all_data = np.hstack((all_data, np.load(file_id, allow_pickle=True)['dat']))
    logging.info("Dataset loaded successfully - {} sessions".format(all_data.shape[0]))
    return all_data