import tensorflow as tf
import gym
import numpy as np
from gym import wrappers


# GLOBAL SETTINGS
RNG_SEED = 8
ENVIRONMENT = "CartPole-v0"
# ENVIRONMENT = "CartPole-v1
MAX_EPISODES = 10000
HIDDEN_LAYER = True
HIDDEN_SIZE = 8
DISPLAY_WEIGHTS = False  # Help debug weight update
RENDER = True  # Render the generation representative
gamma = 0.99  # Discount per step
alpha = 0.000025  # Learning rate

# Upload to OpenAI
UPLOAD = False
EPISODE_INTERVAL = 100  # Generate a video at this interval
SESSION_FOLDER = "/tmp/CartPole-experiment-1"
API_KEY = ""

# Success Mode (Settings to pass OpenAI's requirement)
SUCCESS_MODE = True
SUCCESS_THRESHOLD = 200
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
returns = tf.placeholder(tf.float32, shape=(None, 1))

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
train = optimizer.minimize(-1.0 * returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for ep in range(MAX_EPISODES):
    obs = env.reset()