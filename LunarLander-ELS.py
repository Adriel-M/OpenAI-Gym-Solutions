# Evolutionary Learning Strategy Implementation
# Learn more from https://blog.openai.com/evolution-strategies/

import gym
import numpy as np
from gym import wrappers

# GLOBAL SETTINGS
RNG_SEED = 8
POPULATION_SIZE = 100  # Population size
GENERATION_LIMIT = 100  # Max number of generations
DISPLAY_WEIGHTS = False  # Help debug weight update
sigma = 0.1  # Noise standard deviation
alpha = 0.00025  # Learning rate

# Upload to openai?
UPLOAD = False
UPLOAD_GENERATION_INTERVAL = 10  # Generate a video at this interval
SESSION_FOLDER = "/tmp/LunarLander-experiment-1"
API_KEY = ""

# Success Mode (Settings to pass OpenAI's requirement)
SUCCESS_MODE = True
SUCCESS_THRESHOLD = 200
CONSECUTIVE_TARGET = 100


def record_interval(n):
    global UPLOAD_GENERATION_INTERVAL
    global POPULATION_SIZE
    episode_interval = (POPULATION_SIZE + 1) * UPLOAD_GENERATION_INTERVAL
    return n % episode_interval == 0


def run_episode(environment, weight):
    obs = environment.reset()
    episode_reward = 0
    done = False
    while not done:
        action = np.matmul(weight.T, obs)
        move = np.argmax(action)
        obs, reward, done, info = environment.step(move)
        episode_reward += reward
    return episode_reward


env = gym.make('LunarLander-v2')
if UPLOAD:
    if SUCCESS_MODE:
        env = wrappers.Monitor(env, SESSION_FOLDER)
    else:
        env = wrappers.Monitor(env, SESSION_FOLDER,
                               video_callable=record_interval)

env.seed(RNG_SEED)
np.random.seed(RNG_SEED)

input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# Initial weights
W = np.zeros((input_size, output_size))

for gen in range(GENERATION_LIMIT):
    # Measure performance per generation
    gen_eval = run_episode(env, W)

    # Success mode enabled
    success_count = 1
    if SUCCESS_MODE:
        track_success = [gen_eval]
        while success_count < CONSECUTIVE_TARGET:
            gen_eval = run_episode(env, W)
            if gen_eval < SUCCESS_THRESHOLD:
                break
            track_success.append(gen_eval)
            success_count += 1
        gen_eval = np.mean(track_success)

    # Keep track of Returns
    R = np.zeros(POPULATION_SIZE)
    # Generate noise
    N = np.random.randn(POPULATION_SIZE, input_size, output_size)
    for j in range(POPULATION_SIZE):
        W_ = W + sigma * N[j]
        R[j] = run_episode(env, W_)

    # Update weights
    # Summation of episode_weight * episode_reward
    weighted_weights = np.matmul(N.T, R).T
    new_W = W + alpha / (POPULATION_SIZE * sigma) * weighted_weights
    if DISPLAY_WEIGHTS:
        print(W)
    W = new_W
    gen_mean = np.mean(R)
    if SUCCESS_MODE:
        out = "Generation {}, Num Success: {}, Success Mean: {}, Population " \
          "Mean: {}".format(gen, success_count, gen_eval, gen_mean)
    else:
        out = \
            "Generation {}, Return: {}, Population Mean: {}".format(gen,
                                                                    gen_eval,
                                                                    gen_mean)
    print(out)

env.close()
if UPLOAD:
    gym.upload(SESSION_FOLDER, api_key=API_KEY)
