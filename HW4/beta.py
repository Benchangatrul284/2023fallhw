import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define the alpha and beta values for the beta distribution
alpha_values = [1.0, 50.0, 100.0]
beta_values = [1.0, 50.0, 100.0]

# Generate x values for the plot
x = np.linspace(0, 1, 100)

# Plot the PDF for each combination of alpha and beta values
for alpha, beta_val in zip(alpha_values, beta_values):
    y = beta.pdf(x, alpha, beta_val)
    plt.plot(x, y, label=f'alpha={alpha}, beta={beta_val}')

# Add labels and a legend to the plot
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()

# Show the plot
plt.show()