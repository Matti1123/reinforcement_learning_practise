"""REINFORCE-Training für beide aufrechten Stäbe."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
import torch

PROJEKT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJEKT))
from modelle.gelenk_policy import GelenkPolicy, beobachtung
from simulation import Doppelpendel


def lernkurve_speichern(rewards, validierung, datei):
    """Speichert Trainings-Rewards und Validierungs-Rewards als Lernkurve."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    datei.parent.mkdir(parents=True, exist_ok=True)
    fenster = 20
    durchschnitt = [
        sum(rewards[max(0, i - fenster + 1):i + 1]) / min(i + 1, fenster)
        for i in range(len(rewards))
    ]
    figure = Figure(figsize=(10, 5), layout="constrained")
    FigureCanvasAgg(figure)
    achse = figure.subplots()
    achse.plot(range(1, len(rewards) + 1), rewards, alpha=0.3, label="Trainings-Reward pro Episode")
    achse.plot(range(1, len(rewards) + 1), durchschnitt, label="Gleitender Durchschnitt (20 Episoden)")
    if validierung:
        achse.plot(*zip(*validierung), marker="o", label="Mittlerer Validierungs-Reward")
    achse.set(title="CartPole mit Gelenk: Lernkurve", xlabel="Trainingsepisode", ylabel="Reward pro Episode")
    achse.grid(alpha=0.2)
    achse.legend()
    figure.savefig(datei, dpi=150)


def episode(policy, seed, training, max_schritte):
    rng = np.random.default_rng(seed)
    env = Doppelpendel(seed)
    env.zustand = rng.uniform(-0.025, 0.025, size=6)
    logs, rewards = [], []
    for _ in range(max_schritte):
        with torch.set_grad_enabled(training):
            verteilung = policy(beobachtung(env.zustand))
            aktion = verteilung.sample() if training else verteilung.probs.argmax()
            if training:
                logs.append(verteilung.log_prob(aktion))
        for _ in range(5):
            env.schritt(1 if aktion.item() else -1)
        x, a, b, _, _, _ = env.zustand
        # Hoher Reward, wenn beide Winkel klein sind; leichten Randabstand belohnen.
        reward = max(0.0, 1.0 - (a * a + b * b) - 0.02 * x * x)
        rewards.append(reward)
        if abs(a) > 0.5 or abs(b) > 0.5 or abs(x) >= 2.39:
            break
    return logs, rewards


def bewerten(policy, seed, anzahl, max_schritte):
    ergebnisse = [episode(policy, seed + i, False, max_schritte)[1] for i in range(anzahl)]
    return float(np.mean([sum(r) for r in ergebnisse])), float(np.mean([len(r) for r in ergebnisse]))


def main():
    parser = argparse.ArgumentParser(description="Doppelpendel mit REINFORCE trainieren")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    if min(args.episodes, args.eval_episodes, args.eval_interval, args.max_steps) < 1 or args.seed < 0:
        parser.error("Episoden und Schritte müssen positiv sein, der Seed darf nicht negativ sein")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    policy = GelenkPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.005)
    datei = PROJEKT / "policies" / f"cartpole_gelenk_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.pt"
    datei.parent.mkdir(exist_ok=True)
    plot_datei = PROJEKT / "cartpole_gelenk" / "outputs" / f"{datei.stem}_lernkurve.png"
    # Derselbe Validierungssatz für jeden Kandidaten, getrennt von Trainings-Seeds.
    eval_seed = args.seed + args.episodes + 10000
    beste_bewertung = -float("inf")
    batch_logs, batch_returns = [], []
    training_rewards, validierung = [], []
    for index in range(args.episodes):
        logs, rewards = episode(policy, args.seed + index, True, args.max_steps)
        training_rewards.append(sum(rewards))
        rueckgaben, wert = [], 0.0
        for reward in reversed(rewards):
            wert = reward + 0.99 * wert
            rueckgaben.append(wert)
        batch_logs.extend(logs)
        batch_returns.extend(reversed(rueckgaben))
        if (index + 1) % 10 == 0 or index + 1 == args.episodes:
            returns = torch.tensor(batch_returns, dtype=torch.float32)
            vorteil = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
            loss = -(torch.stack(batch_logs) * vorteil).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_logs.clear()
            batch_returns.clear()
        if (index + 1) % args.eval_interval == 0 or index + 1 == args.episodes:
            score, dauer = bewerten(policy, eval_seed, args.eval_episodes, args.max_steps)
            validierung.append((index + 1, score))
            print(f"Episode {index + 1}: Test-Reward {score:.2f}, Dauer {dauer:.1f}/{args.max_steps} Schritte", flush=True)
            if score > beste_bewertung:
                beste_bewertung = score
                torch.save(policy.state_dict(), datei)
                print(f"  Beste Policy gespeichert: {datei}", flush=True)
            lernkurve_speichern(training_rewards, validierung, plot_datei)
    print(f"Beste Validierung: {beste_bewertung:.2f}; Seeds {eval_seed} bis {eval_seed + args.eval_episodes - 1}", flush=True)
    print(f"Lernkurve gespeichert: {plot_datei}", flush=True)


if __name__ == "__main__":
    main()
