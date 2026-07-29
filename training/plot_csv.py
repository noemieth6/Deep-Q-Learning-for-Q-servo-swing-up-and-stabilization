import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

def init_logger(filename="training_log.csv"):

    # Supprime l'ancien fichier s'il existe
    if os.path.exists(filename):
        os.remove(filename)

    # Création d'un nouveau fichier vide
    f = open(filename, "w", newline="")

    writer = csv.writer(f)

    writer.writerow([
        "episode",
        "step",
        "time",
        "alpha",
        "alpha_error",
        "theta",
        "alpha_dot",
        "theta_dot",
        "action_voltage",
        "reward",
        "episode_reward",
        "best_reward",
        "done"
    ])

    return f, writer


def plot_reward_per_episode(filename="training_log.csv"):

    # Lecture du fichier
    data = pd.read_csv(filename)

    # Garder seulement le dernier reward cumulé de chaque épisode
    episode_rewards = (
        data.groupby("episode")["episode_reward"]
        .max()
        .reset_index()
    )
    episode_rewards = (
    data.groupby("episode")
    .agg({
        "episode_reward": "max",
        "best_reward": "max"
    })
    .reset_index()
)

    # Lissage optionnel (moyenne glissante)
    window = 50
    episode_rewards["reward_smooth"] = (
        episode_rewards["episode_reward"]
        .rolling(window)
        .mean()
    )



    # Plot
    plt.figure(figsize=(10,5))

    plt.plot(
        episode_rewards["episode"],
        episode_rewards["episode_reward"],
        alpha=0.3,
        label="Reward épisode"
    )

    plt.plot(
        episode_rewards["episode"],
        episode_rewards["reward_smooth"],
        linewidth=2,
        label=f"Moyenne glissante ({window})"
    )

    plt.plot(
        episode_rewards["episode"],
        episode_rewards["best_reward"],
        linewidth=2,
        label=f"best reward"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward total")
    plt.title("Evolution du reward pendant l'entraînement")
    plt.grid(True)
    plt.legend()

    plt.show()

# plot_reward_per_episode()