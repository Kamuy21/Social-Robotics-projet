import gymnasium as gym
from stable_baselines3 import SAC  # ou TD3, PPO si tu veux tester

env = gym.make("Pusher-v5")

# Entraînement de la politique experte
model_expert = SAC(
    "MlpPolicy",
    env,
    verbose=1,
)
model_expert.learn(total_timesteps=200_000)  # à adapter selon ton temps de calcul

# Sauvegarde
model_expert.save("pusher_expert_sac")
env.close()
