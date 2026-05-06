import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(filepath: str):
    """Load dataset and split into train/test."""
    df = pd.read_csv(filepath)

    train_df, test_df = train_test_split(df, test_size=0.3)

    stan_data = {
        "N": len(train_df),
        "cost": train_df["cost"].values,
        "success": train_df["success"].values,
        "emotion": train_df["emotion"].values,
        "y": train_df["y"].astype(int).values,
    }
    return train_df, test_df, stan_data

def run_model(stan_data: dict):
    """Compile and sample model."""
    model = CmdStanModel(stan_file="model.stan")

    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_sampling=1000,
        iter_warmup=1000,
        seed=42,
    )
    return fit

def diagnostics(fit):
    """Print convergence diagnostics."""
    summary = fit.summary()
    print("\n=== Diagnostics ===")
    print(summary[["R_hat", "ESS_bulk"]])

def get_posterior(fit):
    """Extract posterior."""
    posterior = fit.draws_pd()
    print("\n=== Posterior Means ===")
    print(posterior.mean())
    return posterior

def compute_accuracy(test_df, posterior):
    """Compute test accuracy."""
    beta_0 = posterior["beta_0"].mean()
    beta_cost = posterior["beta_cost"].mean()
    beta_success = posterior["beta_success"].mean()
    beta_emotion = posterior["beta_emotion"].mean()

    logits = (
        beta_0
        + beta_cost * test_df["cost"]
        + beta_success * test_df["success"]
        + beta_emotion * test_df["emotion"]
    )

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    accuracy = np.mean(preds == test_df["y"])
    print(f"\nTest Accuracy: {accuracy:.3f}")
    return preds

def plot_posteriors(posterior):
    """Posterior histograms."""
    for param in ["beta_cost", "beta_success", "beta_emotion"]:
        plt.hist(posterior[param], bins=30, density=True)
        plt.title(f"Posterior of {param}")
        plt.xlabel(param)
        plt.ylabel("Density")

        plt.savefig(os.path.join(OUTPUT_DIR, f"{param}.png"))
        plt.clf()

def plot_coefficients(posterior):
    """Coefficient bar chart."""
    params = ["beta_0", "beta_cost", "beta_success", "beta_emotion"]
    means = posterior[params].mean()
    means.plot(kind="bar")
    plt.title("Coefficient Means")
    plt.savefig(os.path.join(OUTPUT_DIR, "coefficients.png"))
    plt.clf()

def plot_probability_curve(posterior):
    """Cost vs probability curve."""
    x_vals = np.linspace(0, 1, 100)
    beta_0 = posterior["beta_0"].mean()
    beta_cost = posterior["beta_cost"].mean()
    logit = beta_0 + beta_cost * x_vals
    prob = 1 / (1 + np.exp(-logit))

    plt.plot(x_vals, prob)
    plt.title("Cost vs Probability")
    plt.xlabel("Cost")
    plt.ylabel("Probability")
    plt.savefig(os.path.join(OUTPUT_DIR, "cost_curve.png"))
    plt.clf()

def plot_confusion(test_df, posterior):
    """Confusion matrix on test data."""
    beta_0 = posterior["beta_0"].mean()
    beta_cost = posterior["beta_cost"].mean()
    beta_success = posterior["beta_success"].mean()
    beta_emotion = posterior["beta_emotion"].mean()

    logits = (
        beta_0
        + beta_cost * test_df["cost"]
        + beta_success * test_df["success"]
        + beta_emotion * test_df["emotion"]
    )

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)
    cm = confusion_matrix(test_df["y"], preds)

    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.savefig(os.path.join(OUTPUT_DIR, "confusion.png"))
    plt.clf()

def plot_noise(test_df, posterior):
    """Noise sensitivity analysis."""
    noise_levels = np.linspace(0, 1, 10)
    accuracies = []
    beta_0 = posterior["beta_0"].mean()
    beta_cost = posterior["beta_cost"].mean()
    beta_success = posterior["beta_success"].mean()
    beta_emotion = posterior["beta_emotion"].mean()

    for noise in noise_levels:
        noisy_cost = test_df["cost"] + np.random.normal(0, noise, len(test_df))

        logits = (
            beta_0
            + beta_cost * noisy_cost
            + beta_success * test_df["success"]
            + beta_emotion * test_df["emotion"]
        )

        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        acc = np.mean(preds == test_df["y"])
        accuracies.append(acc)

    plt.plot(noise_levels, accuracies)
    plt.title("Noise vs Accuracy")
    plt.xlabel("Noise")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(OUTPUT_DIR, "noise.png"))
    plt.clf()

def main():
    train_df, test_df, stan_data = load_data("data.csv")
    fit = run_model(stan_data)
    diagnostics(fit)
    posterior = get_posterior(fit)

    compute_accuracy(test_df, posterior)
    plot_posteriors(posterior)
    plot_coefficients(posterior)
    plot_probability_curve(posterior)
    plot_confusion(test_df, posterior)
    plot_noise(test_df, posterior)
    print(f"\nDone. Plots saved in ./{OUTPUT_DIR}/")

if __name__ == "__main__":
    main()