import argparse
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import gymnasium as gym
import torch

from policy import Policy
from plotting import plot_rewards
from reinforcement import evaluate_policy, train_reinforce


def main():
    parser = argparse.ArgumentParser(description="CartPole mit REINFORCE lernen")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument(
        "--load", nargs="+", type=Path, metavar="DATEI",
        help="Gespeicherte Policies laden und unter gleichen Bedingungen vergleichen",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes muss mindestens 1 sein")

    torch.manual_seed(42)

    if args.load:
        env = gym.make("CartPole-v1")
        try:
            for path in args.load:
                policy = Policy()
                try:
                    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                except (OSError, RuntimeError, ValueError) as error:
                    parser.error(f"Policy {path} konnte nicht geladen werden: {error}")
                reward, angle = evaluate_policy(env, policy, episodes=20, seed=10_000)
                print(f"{path}: Reward {reward:.1f}, Winkelabweichung {angle:.2f}°")
        finally:
            env.close()
        return

    # Policy (policy.py): Übersetzt vier Observations in Wahrscheinlichkeiten
    # für links und rechts. Ihre Gewichte werden beim Training gelernt.
    policy = Policy()
    env = gym.make("CartPole-v1")
    try:
        # evaluate_policy (reinforcement.py): Misst Reward und Winkelabweichung
        # ohne Lernen. Gleiche Test-Seeds erlauben einen Vorher-Nachher-Vergleich.
        before = evaluate_policy(env, policy, episodes=20, seed=10_000)
        print(f"Vor Training: Reward {before[0]:.1f}, Winkelabweichung {before[1]:.2f}°")

        # train_reinforce (reinforcement.py): Sammelt Episoden und passt die
        # Policy anhand der Rewards an. Das Training läuft ohne Fenster.
        rewards = train_reinforce(env, policy, episodes=args.episodes, seed=42)

        after = evaluate_policy(env, policy, episodes=20, seed=10_000)
        print(f"Nach Training: Reward {after[0]:.1f}, Winkelabweichung {after[1]:.2f}°")
    finally:
        env.close()

    # Jede Trainingsausführung erhält einen eigenen Dateinamen.
    policy_dir = Path("policies")
    policy_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_path = policy_dir / f"policy_{timestamp}_{uuid4().hex[:8]}.pt"
    torch.save(policy.state_dict(), policy_path)
    print(f"Policy gespeichert: {policy_path}")

    # plot_rewards (plotting.py): Speichert den Reward-Verlauf und einen
    # gleitenden Durchschnitt über 20 Episoden als Bild, auch ohne Fenster.
    plot_path = plot_rewards(rewards)
    print(f"Lernkurve gespeichert: {plot_path}")

    if not args.no_render:
        env = gym.make("CartPole-v1", render_mode="human")
        try:
            reward, angle = evaluate_policy(env, policy, episodes=1, seed=20_000)
            print(f"Demo: Reward {reward:.0f}, Winkelabweichung {angle:.2f}°")
        finally:
            env.close()


if __name__ == "__main__":
    main()
