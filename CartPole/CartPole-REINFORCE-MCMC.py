# REINFORCE: Monte Carlo Policy Gradient Implementation
# Learn more from Reinforcement Learning: An Introduction (p271)
# by Sutton & Barto

import tensorflow as tf
import gym
import numpy as np
from gym import wrappers


# GLOBAL SETTINGS
RNG_SEED = 8
ENVIRONMENT = "CartPole-v0"
# ENVIRONMENT = "CartPole-v1"
MAX_EPISODES = 1000
HIDDEN_LAYER = True
HIDDEN_SIZE = 6
DISPLAY_WEIGHTS = False  # Help debug weight update
RENDER = False  # Render the episode
gamma = 0.99  # Discount per step
alpha = 0.02205  # Learning rate

# Upload to OpenAI
UPLOAD = False
EPISODE_INTERVAL = 50  # Generate a video at this interval
SESSION_FOLDER = "/tmp/CartPole-experiment-1"
API_KEY = ""

CONSECUTIVE_TARGET = 100


def record_interval(n):
    global EPISODE_INTERVAL
    return n % EPISODE_INTERVAL == 0


env = gym.make(ENVIRONMENT)
if UPLOAD:
    env = wrappers.Monitor(env, SESSION_FOLDER, video_callable=record_interval)

env.seed(RNG_SEED)
np.random.seed(RNG_SEED)
tf.set_random_seed(RNG_SEED)

input_size = env.observation_space.shape[0]
try:
    output_size = env.action_space.shape[0]
except AttributeError:
    output_size = env.action_space.n

# Tensorflow network setup
x = tf.placeholder(tf.float32, shape=(None, input_size))
y = tf.placeholder(tf.float32, shape=(None, 1))
expected_returns = tf.placeholder(tf.float32, shape=(None, 1))

w_init = tf.contrib.layers.xavier_initializer()
if HIDDEN_LAYER:
    hidden_W = tf.get_variable("W1", shape=[input_size, HIDDEN_SIZE],
                               initializer=w_init)
    hidden_B = tf.Variable(tf.zeros(HIDDEN_SIZE))
    dist_W = tf.get_variable("W2", shape=[HIDDEN_SIZE, output_size],
                             initializer=w_init)
    dist_B = tf.Variable(tf.zeros(output_size))
    hidden = tf.nn.elu(tf.matmul(x, hidden_W) + hidden_B)
    dist = tf.tanh(tf.matmul(hidden, dist_W) + dist_B)
else:
    dist_W = tf.get_variable("W1", shape=[input_size, output_size],
                             initializer=w_init)
    dist_B = tf.Variable(tf.zeros(output_size))
    dist = tf.tanh(tf.matmul(x, dist_W) + dist_B)

dist_soft = tf.nn.log_softmax(dist)
dist_in = tf.matmul(dist_soft, tf.Variable([[1.], [0.]]))
pi = tf.contrib.distributions.Bernoulli(dist_in)
pi_sample = pi.sample()
log_pi = pi.log_prob(y)

optimizer = tf.train.RMSPropOptimizer(alpha)
train = optimizer.minimize(-1.0 * expected_returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def run_episode(environment, render=False):
    raw_reward = 0
    discounted_reward = 0
    cumulative_reward = []
    discount = 1.0
    states = []
    actions = []
    obs = environment.reset()
    done = False
    while not done:
        states.append(obs)
        cumulative_reward.append(discounted_reward)
        if render:
            obs.render()
        action = sess.run(pi_sample, feed_dict={x: [obs]})[0]
        actions.append(action)
        obs, reward, done, info = env.step(action[0])
        raw_reward += reward
        if reward > 0:
            discounted_reward += reward * discount
        else:
            discounted_reward += reward
        discount *= gamma
    return raw_reward, discounted_reward, cumulative_reward, states, actions


def display_weights(session):
    global HIDDEN_LAYER
    if HIDDEN_LAYER:
        w1 = session.run(hidden_W)
        b1 = session.run(hidden_B)
        w2 = session.run(dist_W)
        b2 = session.run(dist_B)
        print(w1, b1, w2, b2)
    else:
        w1 = session.run(dist_W)
        b1 = session.run(dist_B)
        print(w1, b1)


returns = []
for ep in range(MAX_EPISODES):
    raw_G, discounted_G, cumulative_G, ep_states, ep_actions = \
        run_episode(env, RENDER and not UPLOAD)
    expected_R = np.transpose([discounted_G - np.array(cumulative_G)])
    sess.run(train, feed_dict={x: ep_states, y: ep_actions,
                               expected_returns: expected_R})
    if DISPLAY_WEIGHTS:
        display_weights(sess)
    returns.append(raw_G)
    returns = returns[-CONSECUTIVE_TARGET:]
    mean_returns = np.mean(returns)
    msg = "Episode: {}, Return: {}, Last {} returns mean: {}"
    msg = msg.format(ep, raw_G, CONSECUTIVE_TARGET, mean_returns)
    print(msg)

env.close()
if UPLOAD:
    gym.upload(SESSION_FOLDER, api_key=API_KEY)
