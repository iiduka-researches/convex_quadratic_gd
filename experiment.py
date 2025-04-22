import numpy as np
import csv
from typing import Callable
import os

# matrix X
dim = 100
X = np.diag(np.arange(1, dim + 1))


# function difinition
def f(theta):
    return 0.5 * theta @ X @ theta


def grad_f(theta):
    return X @ theta


# scheduler
def constant_lr_schedule(eta_0):
    return lambda t: eta_0


def decay_schedule(eta_max):
    return lambda t: eta_max / np.sqrt(t + 1)


def linear_decay_schedule(eta_max, T):
    return lambda t: eta_max * (1 - t / T)


def cosine_schedule(eta_max, T):
    return lambda t: 0.5 * eta_max * (1 + np.cos(np.pi * t / T))


def exp_increase_schedule(eta, r):
    return lambda t: eta * r**t


def linear_increase_schedule(eta_max, T):
    return lambda t: eta_max * t / T


def warmup_schedule(r, T_w, eta_max):
    return lambda t: eta_max / r**T_w * r ** t if t <= T_w else eta_max


# experiment function
def run_experiment(schedule_fn: Callable[[int], float], T: int, n_trials: int = 10):
    f_vals = np.zeros((n_trials, T + 1))
    grad_norms = np.zeros((n_trials, T + 1))
    etas = np.zeros((n_trials, T + 1))

    for trial in range(n_trials):
        theta = np.random.randn(dim)
        for t in range(T + 1):
            f_vals[trial, t] = f(theta)
            grad = grad_f(theta)
            grad_norms[trial, t] = np.linalg.norm(grad)
            if t < T:
                eta_t = schedule_fn(t)
                etas[trial, t] = eta_t
                theta -= eta_t * grad

    return f_vals, grad_norms, etas


def save_combined_csv(t_values, f_values, grad_values, eta_values, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "f_theta", "grad_f", "eta_t"])
        for t in range(len(t_values)):
            writer.writerow([t, f_values[t], grad_values[t], eta_values[t]])


if __name__ == "__main__":
    T = 1000  # step size
    trials = 10
    output_dir = "output/constant2"
    os.makedirs(output_dir, exist_ok=True)

    # Choose learning rate schedule
    scheduler = constant_lr_schedule(eta_0=0.01)

    f_vals, grad_norms, etas = run_experiment(scheduler, T=T, n_trials=trials)

    # Save each trial data
    for i in range(trials):
        save_combined_csv(
            t_values=range(T + 1),
            f_values=f_vals[i],
            grad_values=grad_norms[i],
            eta_values=np.append(etas[i], np.nan),  # pad eta to match length
            filename=f"{output_dir}/trial_{i + 1}.csv"
        )

    # Save averaged data
    f_avg = f_vals.mean(axis=0)
    grad_avg = grad_norms.mean(axis=0)
    eta_avg = etas.mean(axis=0)
    save_combined_csv(
        t_values=range(T + 1),
        f_values=f_avg,
        grad_values=grad_avg,
        eta_values=np.append(eta_avg, np.nan),
        filename=f"{output_dir}/average.csv"
    )
