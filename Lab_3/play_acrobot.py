import os
import gymnasium as gym
import numpy as np

# Файл моделі шукаємо поруч зі скриптом, з якої б папки його не запускали
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acrobot_q_learning.npz")
N_EPISODES = 3 # скільки епізодів показати

# 1. Завантажуємо навчену модель
data = np.load(MODEL_FILE)
q_table = data["q_table"]
N_BINS = data["n_bins"]
LOW, HIGH = data["low"], data["high"]
BIN_EDGES = [np.linspace(LOW[i], HIGH[i], N_BINS[i] + 1)[1:-1] for i in range(len(N_BINS))]
print(f"Модель завантажено з {os.path.basename(MODEL_FILE)}, форма Q-таблиці: {q_table.shape}")


# 2. Та сама дискретизація, що й під час навчання
def discretize(obs):
    cos1, sin1, cos2, sin2, vel1, vel2 = obs
    features = [np.arctan2(sin1, cos1), np.arctan2(sin2, cos2), vel1, vel2]
    return tuple(int(np.digitize(features[i], BIN_EDGES[i])) for i in range(len(N_BINS)))


# 3. Граємо, вибираючи найкращу дію з Q-таблиці
env = gym.make("Acrobot-v1", render_mode="human")
for episode in range(N_EPISODES):
    obs, info = env.reset()
    steps = 0
    done = False
    while not done:
        action = int(np.argmax(q_table[discretize(obs)]))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated
    result = "мети досягнуто" if terminated else "не встиг за 500 кроків"
    print(f"Епізод {episode + 1}: {steps} кроків, {result}")
env.close()
