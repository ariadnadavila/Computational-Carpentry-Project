# Simulation/Modeling
# 1. Monte Carlo π estimation (intro exercise)
#   a.  Write a function estimate_pi(N) that: (10 points)
#       Shoots N random points uniformly inside a square of side 2 (x and y from -1 to 1). 
#       Counts how many points fall inside the unit circle (distance ≤ 1 from origin). 
#       Using that, calculate π.
#       Hint: What is the probability of a randomly placed point being in the circle?
#   b.  Plot the convergence of your estimate versus N to visualize accuracy improvement.(5 points)

import numpy as np
import math
import matplotlib.pyplot as plt

# 1a) Function to estimate π using N random points in a square [-1,1] x [-1,1]
def estimate_pi(N: int) -> float:
     # Generate N random points
    x = np.random.uniform(-1, 1, N)
    y = np.random.uniform(-1, 1, N)
    # Checks conditions for all points at once and returns True/False array 
    inside_circle = (x**2 + y**2) <= 1
    # fraction of points inside the circle
    # Probability of being inside = area(circle)/area(square) = π/4 
    # Hence by multiplying by 4, we get an estimate of π
    return float(4 * np.mean(inside_circle))

# Example: compute π with 10000 points
print("Estimate with N=10000:", estimate_pi(10000))

# Plot convergence
Ns = np.unique(np.logspace(1, 6, num=200, dtype=int))  # generate 200 unique integer numbers spaced logarithmically between 10 → 1,000,000
estimates = [estimate_pi(N) for N in Ns] #For each value of N in Ns, compute the pi estimate with that many points. Collect results in a list estimates.

plt.figure(figsize=(7,4))
plt.axhline(math.pi, color="red", linestyle="--", label="True π")
plt.plot(Ns, estimates, label="Estimate", linewidth=1.5) 
plt.xscale("log")
plt.xlabel("N (number of random points)")
plt.ylabel("π estimate")
plt.title("Monte Carlo π Estimation")
plt.legend()
plt.show()

# 1b)  Plot of convergence of estimate versus N to visualize accuracy improvement

# Function to estimate reaction probability
def estimate_reaction_probability(M: int, threshold: float, distribution="uniform") -> float:
    if distribution == "uniform":
        # Energies uniformly distributed between 0 and 10
        energies = np.random.uniform(0, 10, M)
    elif distribution == "normal":
        #Energies normally distributed (mean=5, std=2)
        energies = np.random.normal(loc=5, scale=2, size=M)
    else:
        raise ValueError("Unknown distribution type")
        #If the distribution argument is not normal -> stops program and shows error.
    
    reactions = np.sum(energies > threshold)  # count collisions above threshold
    # Reaction probability = (number of reactions) ÷ (total collisions).
    return float(reactions / M)

# Example: simulate with large M
print("Uniform dist, M=100000:", estimate_reaction_probability(100000, threshold=6, distribution="uniform"))
print("Normal dist,  M=100000:", estimate_reaction_probability(100000, threshold=6, distribution="normal"))

# Plot convergence
Ms = np.unique(np.logspace(2, 5, num=200, dtype=int))  # generate 200 unique integer numbers spaced logarithmically between 10 → 1,000,000
estimates_uniform = [estimate_reaction_probability(M, threshold=6, distribution="uniform") for M in Ms]
estimates_normal  = [estimate_reaction_probability(M, threshold=6, distribution="normal") for M in Ms]

plt.figure(figsize=(7,4))
plt.plot(Ms, estimates_uniform, label="Uniform distribution", linewidth=1.5) 
plt.plot(Ms, estimates_normal, label="Normal distribution", linewidth=1.5) 
plt.xscale("log")
plt.xlabel("M (number of collisions)")
plt.ylabel("Estimated reaction probability")
plt.title("Monte Carlo Simulation of Molecular Collisions")
plt.legend()
plt.show()
