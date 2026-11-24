"""Testet eine gespeicherte CartPole-Policy mit mehreren Startbedingungen."""

import argparse
import math
from pathlib import Path
from statistics import mean

import gymnasium as gym
import torch

from policy import Policy


def main():
    parser = argparse.ArgumentParser(description="Gespeicherte CartPole-Policy testen")
    parser.add_argument(
        "policy", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "policies/policy_20261124_214753_d3e62823.pt",
        help="Pfad zur Policy-Datei (Standard: Policy von 21:47)",
    )
    parser.add_argument("--episodes", type=int, default=0, help="Anzahl Testepisoden (Standard: 50)")
    parser.add_argument("--seed", type=int, default=10000, help="Erster Test-Seed (Standard: 10000)")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes muss mindestens 1 sein")

    policy = Policy()
    try:
        policy.load_state_dict(torch.load(args.policy, map_location="cpu", weights_only=True))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(f"Policy konnte nicht geladen werden: {error}")
    policy.eval()

    env = gym.make("CartPole-v1")
    steps_per_episode = []
    mean_distances = []
    mean_angles = []
    successes = 0
    try:
        with torch.no_grad():
            for episode in range(args.episodes):
                observation, _ = env.reset(seed=args.seed + episode)
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

        seconds_per_step = env.unwrapped.tau
        print(f"Policy: {args.policy}")
        print(f"Testepisoden: {args.episodes} (Seeds {args.seed} bis {args.seed + args.episodes - 1})")
        print(f"Durchschnittliche Dauer: {mean(steps_per_episode):.1f} Schritte "
              f"({mean(steps_per_episode) * seconds_per_step:.2f} simulierte Sekunden)")
        print(f"Kürzeste Episode: {min(steps_per_episode)} Schritte")
        print(f"Erfolgsrate (Zeitlimit ohne Scheitern): {successes}/{args.episodes} "
              f"({100 * successes / args.episodes:.1f} %)")
        print(f"Durchschnittlicher Abstand zur Mitte: {mean(mean_distances):.3f} "
              f"(Grenze: ±{env.unwrapped.x_threshold:g})")
        print(f"Durchschnittliche Winkelabweichung: {mean(mean_angles):.2f}°")
    finally:
        env.close()


if __name__ == "__main__":
    main()
