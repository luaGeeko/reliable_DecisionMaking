import copy
import load_data
import split_test_and_training_data
import numpy as np
import matplotlib.pyplot as plt


def get_sample_cov_matrix(X):
    """
    Returns the sample covariance matrix of data X.

    Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable

    Returns:
    (numpy array of floats)   : Covariance matrix
    """

    X = X - np.mean(X, 0)
    cov_matrix = 1 / X.shape[0] * np.matmul(X.T, X)
    return cov_matrix


def sort_evals_descending(evals, evectors):
    """
    Sorts eigenvalues and eigenvectors in decreasing order. Also aligns first two
    eigenvectors to be in first two quadrants (if 2D).

    Args:
    evals (numpy array of floats)    :   Vector of eigenvalues
    evectors (numpy array of floats) :   Corresponding matrix of eigenvectors
                                         each column corresponds to a different
                                         eigenvalue

    Returns:
    (numpy array of floats)          : Vector of eigenvalues after sorting
    (numpy array of floats)          : Matrix of eigenvectors after sorting
    """

    index = np.flip(np.argsort(evals))
    evals = evals[index]
    evectors = evectors[:, index]
    if evals.shape[0] == 2:
        if np.arccos(np.matmul(evectors[:, 0],
                               1 / np.sqrt(2) * np.array([1, 1]))) > np.pi / 2:
          evectors[:, 0] = -evectors[:, 0]
        if np.arccos(np.matmul(evectors[:, 1],
                               1 / np.sqrt(2)*np.array([-1, 1]))) > np.pi / 2:
          evectors[:, 1] = -evectors[:, 1]

    return evals, evectors


def change_of_basis(X, W):
    """
    Projects data onto a new basis.

    Args:
    X (numpy array of floats) : Data matrix each column corresponding to a
                                different random variable
    W (numpy array of floats) : new orthonormal basis columns correspond to
                                basis vectors

    Returns:
    (numpy array of floats)   : Data matrix expressed in new basis
    """

    Y = np.matmul(X, W)

    return Y

def get_variance_explained(evals):
    """
    Plots eigenvalues.
    Args:
    (numpy array of floats) : Vector of eigenvalues
    Returns:
    Nothing.
    """

    # cumulatively sum the eigenvalues
    csum = np.cumsum(evals)
    # normalize by the sum of eigenvalues
    variance_explained = csum / np.sum(evals)

    return variance_explained


def reconstruct_data(score, evectors, X_mean, K):
    """
    Reconstruct the data based on the top K components.
    Args:
    score (numpy array of floats)    : Score matrix
    evectors (numpy array of floats) : Matrix of eigenvectors
    X_mean (numpy array of floats)   : Vector corresponding to data mean
    K (scalar)                       : Number of components to include
    Returns:
    (numpy array of floats)          : Matrix of reconstructed data
    """

    # Reconstruct the data from the score and eigenvectors
    X_reconstructed = np.matmul(score[:, :K], evectors[:, :K].T) + X_mean

    return X_reconstructed

def pca(X):
    """
    Performs PCA on multivariate data. Eigenvalues are sorted in decreasing order

    Args:
     X (numpy array of floats) :   Data matrix each column corresponds to a
                                   different random variable

    Returns:
    (numpy array of floats)    : Data projected onto the new basis
    (numpy array of floats)    : Vector of eigenvalues
    (numpy array of floats)    : Corresponding matrix of eigenvectors

    """

    X = X - np.mean(X, 0)
    cov_matrix = get_sample_cov_matrix(X)
    evals, evectors = np.linalg.eigh(cov_matrix)
    evals, evectors = sort_evals_descending(evals, evectors)
    score = change_of_basis(X, evectors)

    return score, evectors, evals


def plot_eigenvalues(evals, limit=True):
    """
    Plots eigenvalues.

    Args:
     (numpy array of floats) : Vector of eigenvalues

    Returns:
    Nothing.

    """
    plt.figure()
    plt.plot(np.arange(1, len(evals) + 1), evals, 'o-k')
    plt.xlabel('Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree plot')
    if limit:
        plt.show()


def plot_variance_explained(variance_explained):
    """
    Plots eigenvalues.

    Args:
    variance_explained (numpy array of floats) : Vector of variance explained
                                                 for each PC

    Returns:
    Nothing.

    """

    plt.figure()
    plt.plot(np.arange(1, len(variance_explained) + 1), variance_explained, '--k')
    plt.xlabel('Number of components')
    plt.ylabel('Variance explained')
    plt.show()


def dimensionality_reduction(data, variance_thresh):
    """
    Adds a column 'reconstructed_data' to each trial with a new spike matrix (time bins x neurons) after dimensionality
    reduction has been performed

    Args:
        data (class) : training or test data
        variance_thresh (float) : the percentage of variance in data to be explained by the components
    Returns:
        data (class), the input data with an additional column
    """
    # plt.figure()
    # loop through sessions
    for session_id in range(len(data)):


        spike_data = data[session_id]['spks']
        print("Session", session_id)

        # reshape 3D neurons x trials x time bins matrix into 2D time bins x neurons matrix
        spike_times_concatenated = np.concatenate(spike_data.T)

        # do pca
        score, evectors, evals = pca(spike_times_concatenated)
        variance_explained = get_variance_explained(evals)

        # make plots
        # ax1 = plt.subplot(5, 8, session_id+1)
        # ax1.plot(np.arange(1, len(variance_explained) + 1), variance_explained, '--k')
        # plot_eigenvalues(evals, limit=True)
        # plot_variance_explained(variance_explained)

        # Reconstruct the data based on components resulting in 90% of the variance explained
        K = np.where(variance_explained >= variance_thresh)[0][0]
        X_reconstructed = reconstruct_data(score, evectors, np.mean(spike_times_concatenated, 0), K)

        # add new column to data with newly reconstructed data
        data[session_id]['reconstructed_data'] = X_reconstructed

    # plt.show()
    return data


def main():
    all_data = load_data.load_mouse_data()
    print("Data loaded")
    # test_data = split_test_and_training_data.get_test_data(all_data)
    # training_data = split_test_and_training_data.get_training_data(all_data)
    train_data, test_data = split_test_and_training_data.train_test_split(train_size=0.8)
    train_data = dimensionality_reduction(train_data, 0.9)

if __name__ == '__main__':
    main()