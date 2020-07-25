import numpy as np
import matplotlib.pylab as plt


def my_gaussian(x_points, mu, sigma):
    """
    from: https://github.com/NeuromatchAcademy/course-content/blob/master/tutorials/W2D1_BayesianStatistics/solutions/W2D1_Tutorial1_Solution_aeeeaedf.py
    Returns normalized Gaussian estimated at points `x_points`, with parameters:
     mean `mu` and std `sigma`

    Args:
        x_points (numpy array of floats): points at which the gaussian is
                                          evaluated
        mu (scalar): mean of the Gaussian
        sigma (scalar): std of the gaussian

    Returns:
        (numpy array of floats) : normalized Gaussian evaluated at `x`
    """
    px = np.exp(- 1/ 2 / sigma ** 2 * (mu - x_points) ** 2)
    px = px / px.sum()  # this is the normalization: this part ensures the sum of
    # the individual probabilities at each `x` add up to one.
    # It makes a very strong assumption though:
    # That the `x_points` cover the big portion of
    # probability mass around the mean.
    # Please think/discuss when this would be a dangerous
    # assumption.
    # E.g.: What do you think will happen to the values on
    # the y-axis
    # if the `x` values (x_point) range from -1 to 8 instead
    # of -8 to 8?
    return px


def get_mixture_of_random_gaussians(number_of_gaussians, x_points):
    mixture = np.zeros(len(x_points))
    for component in range(number_of_gaussians):
        random_mu = np.random.uniform(0, len(x_points))
        random_sigma = np.random.uniform(0, len(x_points)/4)
        new_gauss = my_gaussian(x_points, random_mu, random_sigma)
        new_gauss = new_gauss / np.linalg.norm(new_gauss)
        mixture += new_gauss
    return mixture


def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window


def main():
    mix = get_mixture_of_random_gaussians(15, np.arange(250))
    plt.plot(mix)
    plt.show()



if __name__ == '__main__':
    main()
