import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# ============================================================
# Part 1 — Probability Foundations
# ============================================================

def probability_union(PA, PB, PAB):
    if not (0 <= PA <= 1 and 0 <= PB <= 1 and 0 <= PAB <= 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    return PA + PB - PAB


def conditional_probability(PAB, PB):
    if PB == 0:
        raise ValueError("P(B) cannot be zero.")
    return PAB / PB


def are_independent(PA, PB, PAB, tol=1e-9):
    if not (0 <= PA <= 1 and 0 <= PB <= 1 and 0 <= PAB <= 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    return abs(PAB - (PA * PB)) < tol


def bayes_rule(PBA, PA, PB):
    if PB == 0:
        raise ValueError("P(B) cannot be zero.")
    if not (0 <= PBA <= 1 and 0 <= PA <= 1 and 0 <= PB <= 1):
        raise ValueError("Probabilities must be between 0 and 1.")
    return (PBA * PA) / PB


# ============================================================
# Part 2 — Bernoulli Distribution
# ============================================================

def bernoulli_pmf(x, theta):
    if x not in [0, 1]:
        raise ValueError("x must be 0 or 1.")
    if not (0 <= theta <= 1):
        raise ValueError("theta must be between 0 and 1.")
    return (theta ** x) * ((1 - theta) ** (1 - x))


def bernoulli_theta_analysis(theta_values):
    results = []
    for theta in theta_values:
        if not (0 <= theta <= 1):
            raise ValueError("theta must be between 0 and 1.")
        P1 = bernoulli_pmf(1, theta)
        P0 = bernoulli_pmf(0, theta)
        is_symmetric = abs(theta - 0.5) < 1e-9
        results.append((theta, P0, P1, is_symmetric))
    return results


# ============================================================
# Part 3 — Normal Distribution
# ============================================================

def normal_pdf(x, mu, sigma):
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    coefficient = 1 / (math.sqrt(2 * math.pi) * sigma)
    exponent = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coefficient * exponent


def normal_histogram_analysis(mu_values,
                              sigma_values,
                              n_samples=10000,
                              bins=30):

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if bins <= 0:
        raise ValueError("bins must be positive.")
    if len(mu_values) != len(sigma_values):
        raise ValueError("mu_values and sigma_values must have same length.")

    results = []

    for mu, sigma in zip(mu_values, sigma_values):

        if sigma <= 0:
            raise ValueError("sigma must be positive.")

        samples = np.random.normal(mu, sigma, n_samples)

        sample_mean = np.mean(samples)
        theoretical_mean = mu
        mean_error = abs(sample_mean - theoretical_mean)

        sample_variance = np.var(samples)
        theoretical_variance = sigma ** 2
        variance_error = abs(sample_variance - theoretical_variance)

        # Plot histogram
        plt.figure()
        plt.hist(samples, bins=bins, density=True)

        x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
        y_vals = normal_pdf(x_vals, mu, sigma)
        plt.plot(x_vals, y_vals)

        plt.title(f"Normal Distribution (mu={mu}, sigma={sigma})")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.close()

        results.append(
            (
                mu,
                sigma,
                sample_mean,
                theoretical_mean,
                mean_error,
                sample_variance,
                theoretical_variance,
                variance_error
            )
        )

    return results


# ============================================================
# Part 4 — Uniform Distribution
# ============================================================

def uniform_mean(a, b):
    if a >= b:
        raise ValueError("a must be less than b.")
    return (a + b) / 2


def uniform_variance(a, b):
    if a >= b:
        raise ValueError("a must be less than b.")
    return ((b - a) ** 2) / 12


def uniform_histogram_analysis(a_values,
                               b_values,
                               n_samples=10000,
                               bins=30):

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if bins <= 0:
        raise ValueError("bins must be positive.")
    if len(a_values) != len(b_values):
        raise ValueError("a_values and b_values must have same length.")

    results = []

    for a, b in zip(a_values, b_values):

        if a >= b:
            raise ValueError("a must be less than b.")

        samples = np.random.uniform(a, b, n_samples)

        sample_mean = np.mean(samples)
        theoretical_mean = uniform_mean(a, b)
        mean_error = abs(sample_mean - theoretical_mean)

        sample_variance = np.var(samples)
        theoretical_variance = uniform_variance(a, b)
        variance_error = abs(sample_variance - theoretical_variance)

        # Plot histogram
        plt.figure()
        plt.hist(samples, bins=bins, density=True)

        x_vals = np.linspace(a, b, 1000)
        y_vals = np.ones_like(x_vals) / (b - a)
        plt.plot(x_vals, y_vals)

        plt.title(f"Uniform Distribution (a={a}, b={b})")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.close()

        results.append(
            (
                a,
                b,
                sample_mean,
                theoretical_mean,
                mean_error,
                sample_variance,
                theoretical_variance,
                variance_error
            )
        )

    return results


if __name__ == "__main__":
    print("All functions implemented.")
