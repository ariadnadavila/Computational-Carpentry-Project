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
    return float(4 * np.mean(inside_circle))

# Example: compute π with 10000 points
print("Estimate with N=10000:", estimate_pi(10000))
# Plot convergence
Ns = np.unique(np.logspace(1, 6, num=25, dtype=int))  # 10 → 1,000,000
estimates = [estimate_pi(N) for N in Ns]

plt.figure(figsize=(7,4))
plt.axhline(math.pi, color="red", linestyle="--", label="True π")
plt.plot(Ns, estimates, marker="o", label="Estimate")
plt.xscale("log")
plt.xlabel("N (number of random points)")
plt.ylabel("π estimate")
plt.title("Monte Carlo π Estimation")
plt.legend()
plt.show()