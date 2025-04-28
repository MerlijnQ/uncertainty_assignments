import numpy as np
import matplotlib.pyplot as plt

mean_X_1 = 57.9
SD_X_1 = 2
mean_X_2 = 11.49
SD_X_2 = 6.23
lower_bound = mean_X_2 - SD_X_2 * np.sqrt(3)
upper_bound = 2 * mean_X_2 - lower_bound

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
    x = np.linspace(mean_X_1 - 5*SD_X_1, mean_X_1 + 5*SD_X_1, 1000)

    dx = x[1] - x[0]  # Step size for numerical integration
    integral = []
    for z_val in z:
        integral_i = np.sum(Gaussian(x) * Uniform(z_val-x)) * dx 
        integral.append(integral_i)
    return np.array(integral)

def main():

    X_1 = np.random.normal(mean_X_1, SD_X_1, 1000) # Height of 8 week olds: https://cdn.who.int/media/docs/default-source/child-growth/child-growth-standards/indicators/length-height-for-age/sft_lhfa_boys_z_0_13.pdf?sfvrsn=531f87ad_9

    #a = mu - sigma * np.sqrt(3)
    #b = 2 * mu - a

    X_2 = np.random.uniform(lower_bound, upper_bound, 1000) # Smiling times of 8 week olds: https://openstax.org/books/statistics/pages/5-2-the-uniform-distribution

    y = X_1 + X_2

    plt.hist(y, bins=50, density=True, alpha=0.7, color='g', label='Distribution of concatenated samples')
    plt.show()

    #fZ (z) = ∫ fX (x)fY (z − x)dx -> Convolution of the two distributions

    z = np.linspace(min(y), max(y), 1000)
    plt.plot(z, fZ(z), color='r', label='Convolution of the two distributions')
    plt.xlabel('Sum of variables')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Sum of Gaussian and Uniform Distributions')
    plt.show()



if __name__ == '__main__':
    main()




