import math

import torch


def discounted_returns(rewards, gamma):
    """Berechnet für jeden Schritt die Summe seiner abgezinsten zukünftigen Rewards."""
    returns = []
    total = 0.0
    for reward in reversed(rewards):
        total = reward + gamma * total
        returns.append(total)
    return torch.tensor(list(reversed(returns)), dtype=torch.float32)


def train_reinforce(env, policy, episodes=500, seed=42, gamma=0.99, batch_size=10):
    """Trainiert mit REINFORCE; aktualisiert die Policy nach einer Gruppe von Episoden."""
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    batch_log_probs = []
    batch_returns = []
    recent_rewards = []
    reward_history = []
    policy.train()

    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        log_probs = []
        rewards = []

        while True:
            distribution = policy(observation)
            action = distribution.sample()
            log_probs.append(distribution.log_prob(action))
            observation, reward, terminated, truncated, info = env.step(action.item())
            # Zuerst lernen wir mit dem Standard-Reward: +1 pro Schritt.
            rewards.append(reward)
            if terminated or truncated:
                break

        batch_log_probs.append(torch.stack(log_probs))
        batch_returns.append(discounted_returns(rewards, gamma))
        episode_reward = sum(rewards)
        reward_history.append(episode_reward)
        recent_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward {episode_reward:.0f}")

        if len(batch_returns) == batch_size or episode == episodes - 1:
            returns = torch.cat(batch_returns)
            # Gemeinsame Zentrierung und Skalierung stabilisieren die Updates.
            advantages = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
            log_probs = torch.cat(batch_log_probs)
            loss = -(log_probs * advantages).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"  Durchschnitt dieser Gruppe: {sum(recent_rewards) / len(recent_rewards):.1f}")
            batch_log_probs.clear()
            batch_returns.clear()
            recent_rewards.clear()

    return reward_history


def evaluate_policy(env, policy, episodes=20, seed=10_000):
    """Testet ohne Updates mit der wahrscheinlichsten Aktion; liefert Reward und Winkel."""
    episode_rewards = []
    episode_angles = []
    policy.eval()

    with torch.no_grad():
        for episode in range(episodes):
            observation, info = env.reset(seed=seed + episode)
            total_reward = 0.0
            angle_sum = 0.0
            steps = 0
            while True:
                action = policy(observation).probs.argmax().item()
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                angle_sum += abs(float(observation[2]))
                steps += 1
                if terminated or truncated:
                    break
            episode_rewards.append(total_reward)
            episode_angles.append(math.degrees(angle_sum / steps))

    return sum(episode_rewards) / episodes, sum(episode_angles) / episodes
