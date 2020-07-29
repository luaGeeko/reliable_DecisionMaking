from scipy.optimize import minimize
import numpy as np
import load_data
import split_test_and_training_data


def neg_log_lik_lnp(theta, X, y):
  """Return -loglike for the Poisson GLM model.
  Args:
    theta (1D array): Parameter vector.
    X (2D array): Full design matrix.
    y (1D array): Data values.
  Returns:
    number: Negative log likelihood.
  """
  # Compute the Poisson log likeliood
  # theta = theta.reshape(-1,1)
  rate = np.exp(X @ theta)
  # print(np.shape(X))
  # print(np.shape(theta.reshape(-1,1)))
  # print(np.shape(y))
  # print(np.shape(rate))
  log_lik = y @ np.log(rate) - rate.sum()
  return -log_lik


def fit_lnp(designMatrix, spikes):
  """Obtain MLE parameters for the Poisson GLM.
  Args:
    stim (1D array): Stimulus values at each timepoint
    spikes (1D array): Spike counts measured at each timepoint
    d (number): Number of time lags to use.
  Returns:
    1D array: MLE parameters
  """

  # Build the design matrix
  y = spikes
  constant = np.ones_like(y)
  X = np.column_stack([constant, designMatrix])

  d = np.shape(designMatrix)[1]

  # Use a random vector of weights to start (mean 0, sd .2)
  x0 = np.random.normal(0, .2, (d + 1,1))

  # Find parameters that minmize the negative log likelihood function
  res = minimize(neg_log_lik_lnp, x0, args=(X, y.T))

  return res["x"]



def get_session_choice_neurons(dat,isTrainingTrial):  # dat us one session isTrainingTrial is 0s 1s
  BIN_SIZE_MS = 10;
  RIGHT_CHOICE = -1;
  LEFT_CHOICE = 1;
  MAD_TO_STD_SCALE = 1.4826
  N_STD_THRESHOLD = 4
  RESPONSE_BINS_SHIFTS = range(-5,0)


  spikeData = dat['spks'][:,isTrainingTrial,:]
  responseTimeBin = np.floor(dat['response_time'][isTrainingTrial]*1000/BIN_SIZE_MS).astype(int)
  responseTypePerTrial = dat['response'][isTrainingTrial]

  nTimeBins = np.shape(spikeData)[2]
  # print(nTimeBins)
  nResponseShiftBins = len(RESPONSE_BINS_SHIFTS)

  nTrials = len(responseTimeBin)
  isInRightResponseBinPerShift = np.zeros((nTrials, nTimeBins ,nResponseShiftBins),dtype=np.bool)
  isInLeftResponseBinPerShift = np.zeros((nTrials, nTimeBins ,nResponseShiftBins),dtype=np.bool)
  # print(np.shape(isInLeftResponseBinPerShift))
  for iTrial in range(nTrials):
    currResponseBin = responseTimeBin[iTrial]
    currentResponseType = responseTypePerTrial[iTrial]
    for iShift in range(nResponseShiftBins):
      currenTimeBinIndex = currResponseBin+RESPONSE_BINS_SHIFTS[iShift]
      if (currenTimeBinIndex>=0) and (currenTimeBinIndex<nTimeBins):
          if currentResponseType==RIGHT_CHOICE:
            isInRightResponseBinPerShift[iTrial,currenTimeBinIndex,iShift] = True
          elif currentResponseType==LEFT_CHOICE:
            isInLeftResponseBinPerShift[iTrial,currenTimeBinIndex,iShift] = True

  leftResponseFeaturesPerBin = np.reshape(isInLeftResponseBinPerShift,(nTrials*nTimeBins ,nResponseShiftBins))
  rightResponseFeaturesPerBin = np.reshape(isInRightResponseBinPerShift,(nTrials*nTimeBins ,nResponseShiftBins))
  featuresPerBin = np.concatenate((leftResponseFeaturesPerBin, rightResponseFeaturesPerBin),1)

  nFeatures = np.shape(featuresPerBin)[1]+1
  nUnits = np.shape(spikeData)[0]

  featuresWeightsPerUnit = np.zeros((nUnits,nFeatures))

  for iUnit in range(nUnits):
    spikeDataForNeuron = spikeData[iUnit,:,:]
    neuronSpikesPerBin = np.reshape(spikeDataForNeuron,(spikeDataForNeuron.shape[0]*spikeDataForNeuron.shape[1] ,1))
    #featuresWeightsPerUnit[iUnit,:] = fit_lnp(featuresPerBin[0:nBinsToInclude,:], neuronSpikesPerBin[0:nBinsToInclude,:])
    # print(np.shape(featuresPerBin))
    # print(np.shape(neuronSpikesPerBin))
    featuresWeightsPerUnit[iUnit,:] = fit_lnp(featuresPerBin, neuronSpikesPerBin)
    if (iUnit+1) % 50 == 0:
      print(".")
    else:
      print(".", end = '')



  side1Weights = np.abs(featuresWeightsPerUnit[:,1:1+nResponseShiftBins]);
  side2Weights = np.abs(featuresWeightsPerUnit[:,1+nResponseShiftBins:1+2*nResponseShiftBins]);
  #print(np.shape(side2Weights))

  sideModulationIndex = (side1Weights-side2Weights)
  # plt.hist(sideModulationIndex[:,0])
  # plt.plot(np.median(featuresWeightsPerUnit,0))

  meanLeftRightModulation = np.mean(sideModulationIndex,1)
  thresholdForChoiceNeurons = MAD_TO_STD_SCALE*N_STD_THRESHOLD*\
    np.median(np.abs(meanLeftRightModulation))

  isLeftChoice = meanLeftRightModulation>thresholdForChoiceNeurons
  isRightChoice = meanLeftRightModulation<-thresholdForChoiceNeurons

  return isLeftChoice, isRightChoice


def main():
    data = []

if __name__ == '__main__':
    main()