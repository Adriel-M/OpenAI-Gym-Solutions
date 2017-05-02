# OpenAI-Gym-Solutions
Solutions to OpenAI-Gym environments using various machine learning algorithms.

# Strategies
**Evolutionary Learning Strategy:**
Start with some initial weights and generate weights for each member in the
 population (by adding noise to the current weight). Track performance of 
 each member's performance and update current weight (by a weighted sum). 
 Learn more about Evolutionary Learning Strategies [here](https://blog.openai.com/evolution-strategies/).

**REINFORCE: Monte Carlo Policy Gradient**
Perform gradient ascent after every episode on the weighted sum at time t, 
the probability of taking the particular action and the expected total 
discounted reward following the current policy. A discounted reward is used
to enforce finishing early. Learn more about REINFORCE: Monte Carlo Policy 
Gradient by reading Reinforcement Learning: An Introduction (chapter 13.3) by 
Sutton & Barto [here](http://incompleteideas.net/sutton/book/).

# Evaluations
**LunarLander-v2**

*ELS*
 [1](https://gym.openai.com/evaluations/eval_CNyX7JcbSvepv5eb8wCsKg)
 [2](https://gym.openai.com/evaluations/eval_2EWkOozOSuULmn3cXcb1w)
 [3](https://gym.openai.com/evaluations/eval_Uz5XStCR4m6rpADrhfxg)

**CartPole-v0**

*ELS*
 [1](https://gym.openai.com/evaluations/eval_S86D3W2ZQoagd9lpEArL9g)

*REINFORCE-MCMC*
 [1](https://gym.openai.com/evaluations/eval_7WLhKMsNT02Q32wuHCuQJg)
 [2](https://gym.openai.com/evaluations/eval_CRaAAHeZQ0SFdG4n2hCDOA)

**CartPole-v1**

*ELS*
 [1](https://gym.openai.com/evaluations/eval_L0nIc9FQzKF7pcn60L7A)

*REINFORCE-MCMC*
 [1](https://gym.openai.com/evaluations/eval_kRIqBe9cQguVRnbKsMZDpA)
 [2](https://gym.openai.com/evaluations/eval_jUXHNsl5SCqSAdDFwGGoQ)