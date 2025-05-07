import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_1samp
from scipy.integrate import cumulative_trapezoid, trapezoid


mean_X_1 = 57.9
SD_X_1 = 2
mean_X_2 = 11.49
SD_X_2 = 6.23
lower_bound = mean_X_2 - SD_X_2 * np.sqrt(3)
upper_bound = 2 * mean_X_2 - lower_bound
NUM_OF_SAMPLES = 5000

def Gaussian(x):
    dist = (1/(SD_X_1*np.sqrt(2*np.pi))) * np.exp(-0.5 * ((x - mean_X_1) / SD_X_1)**2)
    return dist

def Uniform(x):
    x = np.asarray(x)  # make sure x is an array
    return np.where((x >= lower_bound) & (x <= upper_bound), 1 / (upper_bound - lower_bound), 0)

    
def fZ(z):
    # Convolution of the two distributions
    # fZ(z) = ∫ fX(x) * fY(z - x) dx
    # where fX is the Gaussian distribution and fY is the uniform distribution
    x = np.linspace(mean_X_1 - 6*SD_X_1, mean_X_1 + 6*SD_X_1, NUM_OF_SAMPLES)

    dx = x[1] - x[0]  # Step size for numerical integration
    integral = []
    for z_val in z:
        integral_i = np.sum(Gaussian(x) * Uniform(z_val-x)) * dx 
        integral.append(integral_i)
    return np.array(integral)

def main():

    np.random.seed(42)

    X_1 = np.random.normal(mean_X_1, SD_X_1, NUM_OF_SAMPLES) # Height of 8 week olds: https://cdn.who.int/media/docs/default-source/child-growth/child-growth-standards/indicators/length-height-for-age/sft_lhfa_boys_z_0_13.pdf?sfvrsn=531f87ad_9

    #a = mu - sigma * np.sqrt(3)
    #b = 2 * mu - a

    X_2 = np.random.uniform(lower_bound, upper_bound, NUM_OF_SAMPLES) # Smiling times of 8 week olds: https://openstax.org/books/statistics/pages/5-2-the-uniform-distribution

    y = X_1 + X_2

    plt.hist(y, bins=50, density=True, alpha=0.7, color='g', label='Distribution of concatenated samples')
    #plt.show()

    #fZ (z) = ∫ fX (x)fY (z − x)dx -> Convolution of the two distributions

    z = np.linspace(min(y) - 1, max(y) + 1, 10000)
    pdf_vals = fZ(z)
    plt.plot(z, pdf_vals, color='r', label='Convolution of the two distributions')
    plt.xlabel('Sum of variables')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Sum of Gaussian and Uniform Distributions')
    plt.show()
    
    # Get statistics and such
    pdf_vals /= trapezoid(pdf_vals, z)
    cdf_vals = cumulative_trapezoid(pdf_vals, z, initial=0)
    cdf_vals /= cdf_vals[-1]
    def cdf_func(val):
        return np.interp(val, z, cdf_vals)    
    
    statistic, p_value = ks_1samp(y, cdf_func)
    print(f"Max deviation (in percent): {statistic}.")
    print(f"p-value: {p_value}.")


if __name__ == '__main__':
    main()




