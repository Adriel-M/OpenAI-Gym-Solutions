import tensorflow as tf
import gym
import numpy as np
from gym import wrappers


# GLOBAL SETTINGS
RNG_SEED = 8
ENVIRONMENT = "CartPole-v0"
# ENVIRONMENT = "CartPole-v1"
MAX_EPISODES = 2000
HIDDEN_LAYER = True
HIDDEN_SIZE = 6
DISPLAY_WEIGHTS = False  # Help debug weight update
RENDER = False  # Render the generation representative
gamma = 0.99  # Discount per step
alpha = 0.02005  # Learning rate

# Upload to OpenAI
UPLOAD = False
EPISODE_INTERVAL = 50  # Generate a video at this interval
SESSION_FOLDER = "/tmp/CartPole-experiment-1"
API_KEY = ""

# Success Mode (Settings to pass OpenAI's requirement)
SUCCESS_MODE = True
SUCCESS_THRESHOLD = 195
# SUCCESS_THRESHOLD = 475
CONSECUTIVE_TARGET = 100


def record_interval(n):
    global EPISODE_INTERVAL
    return n % EPISODE_INTERVAL == 0


env = gym.make(ENVIRONMENT)
if UPLOAD:
    if SUCCESS_MODE:
        env = wrappers.Monitor(env, SESSION_FOLDER)
    else:
        env = wrappers.Monitor(env, SESSION_FOLDER,
                               video_callable=record_interval)

env.seed(RNG_SEED)
np.random.seed(RNG_SEED)

input_size = env.observation_space.shape[0]
try:
    output_size = env.action_space.shape[0]
except AttributeError:
    output_size = env.action_space.n

# Tensorflow network setup
x = tf.placeholder(tf.float32, shape=(None, input_size))
y = tf.placeholder(tf.float32, shape=(None, 1))
expected_returns = tf.placeholder(tf.float32, shape=(None, 1))

if HIDDEN_LAYER:
    hidden_W = tf.Variable(tf.random_normal([input_size, HIDDEN_SIZE],
                                            stddev=0.01))
    hidden_B = tf.Variable(tf.zeros(HIDDEN_SIZE))
    dist_W = tf.Variable(tf.random_normal([HIDDEN_SIZE, output_size],
                                          stddev=0.01))
    dist_B = tf.Variable(tf.zeros(output_size))
    hidden = tf.nn.elu(tf.matmul(x, hidden_W) + hidden_B)
    dist = tf.tanh(tf.matmul(hidden, dist_W) + dist_B)
else:
    dist_W = tf.Variable(tf.random_normal([input_size, output_size],
                                          stddev=0.01))
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
    raw_G = 0
    discounted_G = 0
    cumaltive_G = []
    discount = 1.0
    ep_states = []
    ep_actions = []
    obs = environment.reset()
    done = False
    while not done:
        ep_states.append(obs)
        cumaltive_G.append(discounted_G)
        if render:
            obs.render()
        action = sess.run(pi_sample, feed_dict={x: [obs]})[0]
        ep_actions.append(action)
        obs, reward, done, info = env.step(action[0])
        raw_G += reward
        if reward > 0:
            discounted_G += reward * discount
        else:
            discounted_G += reward
        discount *= gamma
    return raw_G, discounted_G, cumaltive_G, ep_states, ep_actions


returns = []
for ep in range(MAX_EPISODES):
    raw_G, discounted_G, cumalative_G, ep_states, ep_actions = \
        run_episode(env, RENDER and not UPLOAD)
    expected_R = np.transpose([discounted_G - np.array(cumalative_G)])
    sess.run(train, feed_dict={x: ep_states, y: ep_actions,
                               expected_returns: expected_R})
    returns.append(raw_G)
    returns = returns[-CONSECUTIVE_TARGET:]
    mean_returns = np.mean(returns)
    msg = "Episode: {}, Return: {}, Last {} returns mean: {}"
    msg = msg.format(ep, raw_G, CONSECUTIVE_TARGET, mean_returns)
    print(msg)
    if mean_returns > SUCCESS_THRESHOLD:
        break

env.close()
if UPLOAD:
    gym.upload(SESSION_FOLDER, api_key=API_KEY)
