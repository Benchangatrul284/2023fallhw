# Epsilon-Greedy Algorithm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Setting the seed for reproducibility
np.random.seed(42)

# Initialize parameters
n_arms = 3
true_probabilities = [0.3, 0.7, 0.8]
n_rounds = 1000
epsilon = 0.2  # Epsilon for the ε-greedy algorithm

# Initialize counters for wins and trials for each arm
wins = np.zeros(n_arms)
trials = np.zeros(n_arms)

# Lists to keep track of rewards and chosen arms
rewards = []

for _ in range(n_rounds):
    if np.random.rand() < epsilon:
        # Exploration: choose a random arm
        chosen_arm = np.random.choice(n_arms)
    else:
        # Exploitation: choose the best known arm
        # Arm with the highest success rate so far (wins/trials)
        # Avoid division by zero by adding a small value to trials
        success_rates = wins / (trials + 1e-5)
        chosen_arm = np.argmax(success_rates)
        
    # Simulate pulling the chosen arm
    reward = np.random.rand() < true_probabilities[chosen_arm]
    rewards.append(reward)
    wins[chosen_arm] += reward
    trials[chosen_arm] += 1

# Calculate total reward and regret
total_reward = np.sum(rewards)
max_reward = np.max(true_probabilities) * n_rounds
total_regret = max_reward - total_reward

# Calculate estimated probabilities based on wins and trials
estimated_probabilities = wins / trials

# Plotting the estimated probability distributions
x = np.linspace(0, 1, 100)
for i in range(n_arms):
    # Using a beta distribution to plot the probability estimates
    y = beta.pdf(x, a=wins[i]+1, b=trials[i]-wins[i]+1)
    plt.plot(x, y, label=f'Arm {i+1}')

plt.title('Estimated Probabilities after 1000 Rounds with Epsilon-Greedy')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"Total Reward: {total_reward}")
print(f"Total Regret: {total_regret}")
print(f"Number of times each arm was pulled: {trials}")


# UCB Algorithm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import math

# Setting the seed for reproducibility
np.random.seed(42)

# Initialize parameters
n_arms = 3
true_probabilities = [0.3, 0.7, 0.8]
n_rounds = 1000
total_plays = 0  # Total number of plays across all arms
c = 0.5

# Initialize counters for wins and trials for each arm
wins = np.zeros(n_arms)
trials = np.zeros(n_arms)

# Lists to keep track of rewards and chosen arms
rewards = []
chosen_arms = []
rounds_to_plot = [100, 400, 700, 1000]

# Function to calculate the UCB values
def calculate_ucb(total_plays, wins, trials):
    ucb_values = []
    for arm in range(n_arms):
        if trials[arm] > 0:
            average_reward = wins[arm] / trials[arm]
            delta_i = math.sqrt(c * math.log(total_plays) / trials[arm])
            ucb_values.append(average_reward + delta_i)
        else:
            ucb_values.append(float('inf'))  # A very high value to ensure untried arms are selected
    return ucb_values


# UCB Algorithm
for round_number in range(1, n_rounds + 1):
    ucb_values = calculate_ucb(total_plays, wins, trials)
    chosen_arm = np.argmax(ucb_values)
    chosen_arms.append(chosen_arm)
    reward = np.random.rand() < true_probabilities[chosen_arm]
    rewards.append(reward)
    wins[chosen_arm] += reward
    trials[chosen_arm] += 1
    total_plays += 1

    # Plot the confidence bounds at specified rounds
    if round_number in rounds_to_plot:
        plt.figure(figsize=(10, 6))
        arm_rewards = wins / trials
        confidence_bounds = [math.sqrt(c * math.log(total_plays) / trials[arm]) if trials[arm] > 0 else 0 for arm in range(n_arms)]
        
        for arm in range(n_arms):
            plt.bar(arm, arm_rewards[arm], yerr=confidence_bounds[arm], capsize=5, label=f'Arm {arm+1}')

        plt.title(f'UCB Confidence Bounds at Round {round_number}')
        plt.xlabel('Arm')
        plt.ylabel('Estimated Probability')
        plt.xticks(range(n_arms))
        plt.legend()
        plt.show()

# Calculate total reward and regret
total_reward = np.sum(rewards)
max_reward = np.max(true_probabilities) * n_rounds
total_regret = max_reward - total_reward

# Calculate estimated probabilities based on wins and trials
estimated_probabilities = wins / trials

# Plotting the estimated probability distributions
x = np.linspace(0, 1, 100)
for i in range(n_arms):
    # Using a beta distribution to plot the probability estimates
    y = beta.pdf(x, a=wins[i]+1, b=trials[i]-wins[i]+1)
    plt.plot(x, y, label=f'Arm {i+1}')

plt.title('Estimated Probabilities after 1000 Rounds with UCB')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"Total Reward: {total_reward}")
print(f"Total Regret: {total_regret}")
print(f"Number of times each arm was pulled: {trials}")

# Thompson Sampling Algorithm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Setting the seed for reproducibility
np.random.seed(4)

# Initialize parameters
n_arms = 3
true_probabilities = [0.3, 0.7, 0.8]
n_rounds = 1000

# Initialize counters for wins and trials for each arm
wins = np.zeros(n_arms)
trials = np.zeros(n_arms)

# Lists to keep track of rewards and chosen arms
rewards = []
chosen_arms = []

# Thompson Sampling Algorithm
for round_number in range(1, n_rounds + 1):
    sampled_probs = [beta.rvs(a=wins[i]+1, b=trials[i]-wins[i]+1) for i in range(n_arms)]
    chosen_arm = np.argmax(sampled_probs)
    reward = np.random.rand() < true_probabilities[chosen_arm]
    rewards.append(reward)
    chosen_arms.append(chosen_arm)
    wins[chosen_arm] += reward
    trials[chosen_arm] += 1

    # If the current round is in the list of rounds to plot, plot the distribution
    if round_number in rounds_to_plot:
        x = np.linspace(0, 1, 100)
        plt.figure(figsize=(10, 6))
        for i in range(n_arms):
            y = beta.pdf(x, a=wins[i]+1, b=trials[i]-wins[i]+1)
            plt.plot(x, y, label=f'Arm {i+1}')
        plt.title(f'Thompson sampling after {round_number} Rounds')
        plt.xlabel('Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.show()


# Calculate total reward and regret
total_reward = sum(rewards)
max_reward = max(true_probabilities) * n_rounds
total_regret = max_reward - total_reward

print(f"Total Reward: {total_reward}")
print(f"Total Regret: {total_regret}")
print(f"Number of times each arm was pulled: {trials}")





