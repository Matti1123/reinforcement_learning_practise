from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_rewards(rewards, path="outputs/reward_curve.png", window=20):
    """Speichert Episoden-Rewards und ihren gleitenden Durchschnitt als PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    episodes = range(1, len(rewards) + 1)
    # Zu Beginn wird über die bisher verfügbaren Episoden gemittelt.
    averages = [
        sum(rewards[max(0, i - window + 1):i + 1]) / min(i + 1, window)
        for i in range(len(rewards))
    ]

    figure = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ax.plot(episodes, rewards, alpha=0.35, label="Reward pro Episode")
    ax.plot(episodes, averages, linewidth=2, label=f"Gleitender Durchschnitt ({window} Episoden)")
    ax.set(
        title="CartPole: Lernkurve mit REINFORCE",
        xlabel="Trainingsepisode",
        ylabel="Gesamter Reward (Standard-Reward = Anzahl Schritte)",
    )
    ax.axhline(500, color="gray", linestyle="--", label="Zeitlimit: 500 Schritte")
    ax.grid(alpha=0.2)
    ax.legend()
    figure.savefig(path, dpi=150)
    return path
