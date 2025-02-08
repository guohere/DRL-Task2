

## Prompt



## Questions
what is PPO, and what is a typical PPO in RLHF look like in detail, like from starts from giving policy(LM) a prompt, then it produce a output, using Reward model to evaluate the output give a reward, and then, update the policy wiht reward, using PPO?
here the PPO will be used as a loss function or what? as far as i know, DPO is used as a loss function to update the policy directly, in this sense, then what is the difference with DPO and PPO? does sometimes DPO also need a reward model? when the dataset is not specificially containing scores of responses? in this case, does DPO becomes a ppo?




## PPO in detail
Proximal Policy Optimization (PPO)
Why PPO?: PPO is chosen because it is stable, relatively simple, and effective for large-scale RL tasks; it prevents excessively large policy updates that can lead to instability (“policy collapse”).
Process:
Sample dialogues (or text completions) from the policy.
Evaluate them with the reward model.
Compute advantage estimates.
Update the policy within a “proximal” trust region to keep changes modest per iteration.


how to compute advantage estimates in detail?

how does this update the policy works per iteration?



