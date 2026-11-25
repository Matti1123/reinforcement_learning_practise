"""Testet eine gespeicherte CartPole-Policy mit mehreren Startbedingungen."""

import argparse
import math
from pathlib import Path
from statistics import mean

import gymnasium as gym
import torch

from policy import Policy


def test_policy(path, episodes=50, seed=10000):
    """Liefert Kennzahlen, gemittelt über unabhängige Testepisoden."""
    if episodes < 1 or seed < 0:
        raise ValueError("episodes muss positiv und seed nicht negativ sein")
    policy = Policy()
    policy.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    policy.eval()

    env = gym.make("CartPole-v1")
    steps_per_episode = []
    mean_distances = []
    mean_angles = []
    successes = 0
    try:
        with torch.no_grad():
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + episode)
                distances, angles = [], []
                while True:
                    action = policy(observation).probs.argmax().item()
                    observation, _, terminated, truncated, _ = env.step(action)
                    distances.append(abs(float(observation[0])))
                    angles.append(abs(float(observation[2])))
                    if terminated or truncated:
                        break
                steps_per_episode.append(len(distances))
                mean_distances.append(mean(distances))
                mean_angles.append(math.degrees(mean(angles)))
                successes += int(truncated and not terminated)

        return {
            "policy": str(path), "episodes": episodes, "first_seed": seed,
            "mean_steps": mean(steps_per_episode),
            "mean_simulated_seconds": mean(steps_per_episode) * env.unwrapped.tau,
            "shortest_episode_steps": min(steps_per_episode),
            "successes": successes, "success_rate": successes / episodes,
            "mean_distance_from_center": mean(mean_distances),
            "position_limit": env.unwrapped.x_threshold,
            "mean_angle_degrees": mean(mean_angles),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Gespeicherte CartPole-Policy testen")
    parser.add_argument("policy", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "policies/policy_20261124_214753_d3e62823.pt")
    parser.add_argument("--episodes", type=int, default=0, help="Anzahl Testepisoden")
    parser.add_argument("--seed", type=int, default=10000, help="Erster Test-Seed")
    args = parser.parse_args()
    try:
        result = test_policy(args.policy, args.episodes, args.seed)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"Policy: {result['policy']}")
    print(f"Testepisoden: {result['episodes']} (Seeds {args.seed} bis {args.seed + args.episodes - 1})")
    print(f"Durchschnittliche Dauer: {result['mean_steps']:.1f} Schritte ({result['mean_simulated_seconds']:.2f} simulierte Sekunden)")
    print(f"Kürzeste Episode: {result['shortest_episode_steps']} Schritte")
    print(f"Erfolgsrate (Zeitlimit ohne Scheitern): {result['successes']}/{args.episodes} ({result['success_rate']:.1%})")
    print(f"Durchschnittlicher Abstand zur Mitte: {result['mean_distance_from_center']:.3f} (Grenze: ±{result['position_limit']:g})")
    print(f"Durchschnittliche Winkelabweichung: {result['mean_angle_degrees']:.2f}°")


if __name__ == "__main__":
    main()
