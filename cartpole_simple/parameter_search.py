import math

import gymnasium as gym


def run_episode(env, factor, seed):
    observation, info = env.reset(seed=seed)
    episode_reward = 0
    score = 0.0
    angle_sum = 0.0
    steps = 0

    while True:
        pole_angle = observation[2]
        pole_angular_velocity = observation[3]
        control = pole_angle + factor * pole_angular_velocity
        action = 0 if control < 0 else 1

        observation, reward, terminated, truncated, info = env.step(action)
        pole_angle = float(observation[2])
        episode_reward += reward
        score += 1.0 - 10.0 * pole_angle**2
        angle_sum += abs(pole_angle)
        steps += 1

        if terminated or truncated:
            return episode_reward, score, math.degrees(angle_sum / steps)


def search_parameters():
    factors = [0.1, 0.3, 0.5, 1.0]
    episodes = 10
    best_factor = None
    best_score = -math.inf
    env = gym.make("CartPole-v1")

    try:
        for factor in factors:
            results = []
            for seed in range(episodes):
                # Gleiche Startbedingungen für jeden Faktor.
                result = run_episode(env, factor, seed)
                results.append(result)
                print(f"Faktor {factor}, Episode {seed + 1}: Reward {result[0]:.0f}")

            mean_reward = sum(result[0] for result in results) / episodes
            mean_score = sum(result[1] for result in results) / episodes
            mean_angle = sum(result[2] for result in results) / episodes
            print(
                f"Faktor {factor}: Durchschnittlicher Reward {mean_reward:.1f}, "
                f"Winkelabweichung {mean_angle:.2f}°, Bewertung {mean_score:.2f}\n"
            )

            if mean_score > best_score:
                best_score = mean_score
                best_factor = factor
    finally:
        env.close()

    print(f"Bester getesteter Faktor: {best_factor}")
    return best_factor


if __name__ == "__main__":
    best_factor = search_parameters()

    # Den ausgewählten Faktor anschließend im Fenster ansehen.
    env = gym.make("CartPole-v1", render_mode="human")
    try:
        reward, score, mean_angle = run_episode(env, best_factor, seed=100)
        print(f"Demo: Reward {reward:.0f}, Winkelabweichung {mean_angle:.2f}°")
    finally:
        env.close()
