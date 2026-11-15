# The algorithm behind the RL walk: PPO from the basics

[rl_walk.md](rl_walk.md) says *what* we trained (inputs, reward, randomization). This file explains
*how the learning itself works*: reinforcement learning, policy gradients and PPO, each tied to the
numbers in our code ([`sim/rl_walk_train.py`](../sim/rl_walk_train.py)).

Read it top to bottom; each section needs the one before it.

## 1. The problem: an agent in a loop

| RL word | Meaning | In our robot |
|---|---|---|
| **state** $s_t$ | the true situation | everything in the simulator |
| **observation** $o_t$ | what the agent can see of it | 47 numbers (41 without IMU) |
| **action** $a_t$ | what the agent decides | 12 joint targets |
| **reward** $r_t$ | one number: how good was this step | the 15-term score |
| **policy** $\pi$ | the rule that picks actions | the neural network |
| **episode** | one try, from reset to fall or time-out | up to 20 s = 1000 steps |

Every 20 ms: the policy sees $o_t$, picks $a_t$, the simulator moves on, and returns $r_t$ and $o_{t+1}$.

The policy sees an observation, not the full state (it has no idea of the floor friction or its own
speed). That is why removing the IMU matters: it removes part of what the policy can know.

## 2. The goal: the return

The agent should not maximize the reward of *this* step, but the sum of what follows. That sum is the **return**:

$$G_t = r_t + \gamma\, r_{t+1} + \gamma^2 r_{t+2} + \dots$$

$\gamma$ (gamma, the **discount**) is slightly below 1, so far-away rewards count less. We use
$\gamma = 0.99$: a reward 100 steps (2 s) ahead still counts $0.99^{100} \approx 37\,\%$. So the robot
"plans" a few seconds ahead: long enough to care that a bad step now causes a fall one second later.

**Goal of RL: find the policy with the highest expected return.**

## 3. A policy that explores: the Gaussian policy

The network does not output one action. It outputs the **mean** $\mu(o)$ of a bell curve, and the action
is drawn from it:

$$a \sim \mathcal{N}\big(\mu_\theta(o),\ \sigma^2\big)$$

- $\theta$ = the network's weights (about 80,000 numbers per network).
- $\sigma$ = the noise size, one learned number per joint. We start at $e^{-1} \approx 0.37$.

The noise is the *trial* in trial and error: without it the robot would repeat the same motion and never
find out whether a slightly different one is better. As training goes on, $\sigma$ shrinks by itself
(the `std` value in the training log). When we test or deploy, we use the mean only, with no noise.

## 4. Two helpers: value and advantage

**Value** $V(o)$: the return you expect from this situation, if you keep following the current policy.
A second network, the **critic**, learns it. (The policy network is the **actor**; the pair is called
**actor-critic**.)

**Advantage** $A(o, a)$: how much better this action turned out than what was expected here.

$$A(o_t, a_t) = \underbrace{G_t}_{\text{what happened}} - \underbrace{V(o_t)}_{\text{what was expected}}$$

- $A > 0$: the action was better than usual → make it more likely.
- $A < 0$: worse than usual → make it less likely.

Why not use the return directly? Because a return is large whenever the robot is already in a good
situation, whatever it does. Subtracting $V$ leaves only the part that the action itself caused.

## 5. Estimating the advantage: GAE

We only simulate 256 steps before each update, not whole episodes, so $G_t$ is not fully known. Two extremes:

- **One step:** $\delta_t = r_t + \gamma V(o_{t+1}) - V(o_t)$. Uses the critic for everything after one
  step. Low noise, but wrong if the critic is wrong.
- **Many steps:** add up the real rewards as far as we have them. Correct on average, but noisy.

**GAE** (Generalized Advantage Estimation) mixes them with a second factor $\lambda$:

$$\hat{A}_t = \delta_t + (\gamma\lambda)\,\delta_{t+1} + (\gamma\lambda)^2\,\delta_{t+2} + \dots$$

We use $\lambda = 0.95$: mostly real rewards over roughly the next 20 steps, the critic after that.

## 6. The policy gradient: the core idea

How do we change $\theta$ so that good actions become more likely? The **policy gradient theorem** gives
the direction:

$$\nabla_\theta J = \mathbb{E}\big[\ \nabla_\theta \log \pi_\theta(a_t \mid o_t)\ \cdot\ \hat{A}_t\ \big]$$

Read it as: *for every action we tried, push its probability up in proportion to its advantage*
(and down if the advantage is negative). Averaged over 32,768 samples, the noise cancels and what
remains is the direction that improves the walk.

The simplest algorithm (REINFORCE / vanilla policy gradient) takes one step in this direction and
throws the data away. Two problems:

1. **Wasteful:** each sample is used once, and samples are expensive (simulation time).
2. **Unstable:** one step that is too large can ruin the policy. The robot falls at once, the new data
   is all falls, and training may never recover.

## 7. PPO: reuse the data, but don't move too far

**PPO** (Proximal Policy Optimization, Schulman et al. 2017) fixes both. It reuses each batch for
several gradient steps, and limits how far the policy may move away from the one that collected the data.

Define the probability ratio between the new and the old policy for an action we tried:

$$\rho_t = \frac{\pi_\theta(a_t \mid o_t)}{\pi_{\text{old}}(a_t \mid o_t)}$$

$\rho = 1$ means unchanged; $\rho = 1.3$ means the action is now 30 % more likely. PPO maximizes:

$$L^{\text{clip}} = \mathbb{E}\Big[\min\big(\rho_t \hat{A}_t,\ \ \text{clip}(\rho_t,\ 1-\varepsilon,\ 1+\varepsilon)\,\hat{A}_t\big)\Big]$$

with $\varepsilon = 0.2$ (our `clip_range`). In words:

- A good action ($\hat A > 0$) is made more likely, but once it is 20 % more likely, there is no further gain.
- A bad action is made less likely, but once it is 20 % less likely, there is no further gain.
- If the policy moved the *wrong* way, the penalty is not capped, so it gets corrected.

So no single batch can change the behaviour by more than about 20 %. That is the "proximal" (= nearby)
in the name, and it is why PPO is stable enough to be the standard for legged robots.

## 8. The full loss

Each gradient step minimizes three things at once:

$$L = -L^{\text{clip}}\ +\ c_v \cdot \underbrace{\big(V_\theta(o_t) - G_t\big)^2}_{\text{critic learns the return}}\ -\ c_e \cdot \underbrace{H[\pi_\theta]}_{\text{entropy}}$$

- **Critic loss:** makes $V$ predict the returns we actually saw (`value_loss` in the log).
- **Entropy bonus** ($c_e = 0.005$): a small reward for keeping $\sigma$ large, so that exploration does
  not stop too early.

## 9. One update, step by step, with our numbers

1. **Collect:** 128 robots × 256 steps = 32,768 samples with the current policy (about 2 s of computing).
2. **Advantages:** compute $\hat A_t$ with GAE ($\gamma = 0.99$, $\lambda = 0.95$), then normalize them
   to mean 0 and spread 1.
3. **Learn:** shuffle the samples, split them into 4 mini-batches of 8,192, and take one gradient step
   on each (Adam optimizer, learning rate 0.0003). Repeat for 5 passes (**epochs**) = 20 gradient steps.
4. **Throw the data away** and go back to 1. The data belonged to the old policy.

120 M steps = about 3,700 such updates.

PPO is **on-policy**: it can only learn from data collected by the current policy. That is why it needs
so many samples, and why fast parallel simulation matters more than a fast GPU.

## 10. Three practical details that matter

- **Observation normalization.** Each input is rescaled to mean 0 and spread 1 using running statistics
  (`VecNormalize`). Joint speeds and clock values have very different sizes; without this the network
  learns badly. The statistics are saved in `vecnormalize.pkl` and belong to the policy: the policy
  is useless without them.
- **Early termination.** An episode ends when the robot falls, so all later reward is lost. This is a
  strong, simple signal and the first thing learned (`ep_len_mean` rises to 1000).
- **Time-outs are not falls.** When an episode ends at 20 s, the critic's value of the last state is
  added back, so the robot does not learn that "at 20 s the world ends".

## 11. How to read the training log

| Log value | Meaning | Healthy sign |
|---|---|---|
| `ep_len_mean` | steps survived per episode | rises to ~1000 |
| `ep_rew_mean` | return per episode | keeps rising, then flattens |
| `std` | exploration noise $\sigma$ | slowly falls |
| `approx_kl` | how far the policy moved this update | small, about 0.01 |
| `clip_fraction` | share of samples that hit the 20 % limit | about 0.05–0.2 |
| `value_loss` | critic's prediction error | falls, then stays low |
| `explained_variance` | how well the critic predicts returns (1 = perfect) | rises towards 1 |
| `fps` | simulated steps per second | speed only, not quality |

## 12. Where PPO sits among RL algorithms

| Family | Examples | Idea | Why not here |
|---|---|---|---|
| Value-based | Q-learning, DQN | learn the value of each action, pick the best | needs a finite set of actions; ours are 12 continuous numbers |
| Policy gradient | REINFORCE | follow the gradient of section 6 | wasteful and unstable |
| **Actor-critic, on-policy** | **PPO**, TRPO, A2C | policy gradient + critic + limited steps | this is ours |
| Actor-critic, off-policy | SAC, TD3, DDPG | reuse old data from a replay buffer | fewer samples needed, but harder to tune and slower per step with thousands of parallel robots |
| Model-based | MPC, Dreamer | learn or use a model and plan with it | the ZMP walk is a hand-made version of this idea |

## Things worth being able to explain

1. Why is the advantage used instead of the return?
2. What do $\gamma$ and $\lambda$ each control, and what happens at 0 and at 1?
3. What does the clip in PPO prevent, and what would happen without it?
4. Why can PPO not reuse data from an old policy (on-policy)?
5. Why does the policy output a distribution and not one action?
6. Why does the no-IMU policy balance but not follow a turn command? (Hint: section 1.)

## Sources

- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017 (PPO).
- Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2016 (GAE).
- Sutton & Barto, *Reinforcement Learning: An Introduction*, chapter 13 (policy gradients).
- OpenAI *Spinning Up*, the "Intro to Policy Optimization" and PPO pages.
