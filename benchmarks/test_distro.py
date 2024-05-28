import numpy as np
import matplotlib.pyplot as plt

# # Set the mode of the distribution
# mode = 4

# # Set the shape parameter (a) less than 1 for left-skewed distribution
# shape_parameter = 0.5

# # Generate random numbers from gamma distribution
# data = np.random.gamma(shape_parameter, mode, 1000)
# right_data = 20 - data

# # bimodal
# bimodal = np.concatenate([data, right_data])

# # Plot the histogram to visualize the distribution
# plt.hist(bimodal, bins=30, density=True, alpha=0.7, color='blue')
# plt.title('Bimodal Distribution with Mode of 4, 16')
# plt.xlabel('Values')
# plt.ylabel('Probability Density')
# plt.show()
# plt.savefig('bimodal')

def generate_exec_times(distro, num_prompts):
  rng = np.random.default_rng(0)
  if distro == 'N':
    # normal distribution
    return np.abs(rng.normal(loc=11, scale=3, size=(num_prompts,)))
  elif distro == 'U':
    # uniform distribution
    return rng.uniform(low=0.1, high=15, size=(num_prompts,))
  else:
    # Generate random numbers from gamma distribution
    right = np.abs(rng.gamma(shape=0.5, scale=4, size=(num_prompts,)))  # shorter api times
    left = np.abs(20-right)                                             # longer api times
    if distro == 'L':
        return left
    elif distro == 'R':
        return right
    elif distro == 'B':
        return np.concatenate([rng.choice(left, num_prompts//2),
                                rng.choice(right, num_prompts//2)])
    else:
        return ValueError(f'Unsupported distribution: {distro}')
    
np.random.seed(0)
print(generate_exec_times('U', 100))