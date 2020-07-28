import numpy as np
import matplotlib.pylab as plt
import timeit
import scipy.stats as stats

# random seed for reproducibility of experiements with dataset splits
np.random.seed(75)
# FIXED: RuntimeWarning: invalid value encountered in true_divide px = px / px.sum()
def my_gaussian(x_points, mu, sigma):
    """
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
    #px = np.exp(- 1/ 2 / sigma ** 2 * (mu - x_points) ** 2)
    #px = px / px.sum()
    px = stats.norm.pdf(x_points, mu, sigma)
    return px

def get_mixture_of_random_gaussians(number_of_gaussians, x_points):
    mixture = np.zeros(len(x_points))
    for component in range(number_of_gaussians):
        random_mu = np.random.uniform(0, len(x_points))
        random_sigma = np.random.uniform(0, len(x_points)/4)
        new_gauss = my_gaussian(x_points, random_mu, random_sigma)
        #new_gauss = new_gauss / np.linalg.norm(new_gauss)
        mixture += new_gauss
    return mixture

def moving_average(x, window):
    return np.convolve(x, np.ones(window), 'valid') / window

def time_cal(func, *args, **kwargs):
    def inner_func():
        return func(*args, **kwargs)
    return inner_func

def main():
    mix = my_gaussian(np.arange(250), 2, 1)
    plt.plot(mix)
    plt.xlabel("TIme bins")
    plt.ylabel("pdf")
    plt.show()