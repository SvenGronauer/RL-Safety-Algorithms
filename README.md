# RL-Safety-Algorithms

Algorithms for Safe Reinforcement Learning Problems that were tested and 
benchmarked in the 
[Open-Safety-Gym](https://github.com/svengronauer/Open-Safety-Gym).

## Installation

Install this repository with:

```
git clone https://github.com/svengronauer/RL-Safety-Algorithms.git

cd RL-Safety-Algorithms

pip install -e .
```


## Getting Started

Works with every environment that is compatible with the OpenAI Gym interface:

```
python rl_safety_algorithms.train --alg trpo --env CartPole-v0
```

For an open-source framework to benchmark and test safety, we recommend the 
[Open-Safety-Gym](https://github.com/svengronauer/Open-Safety-Gym). To train an
algorithms such as Constrained Policy Optimization, run:

```
python rl_safety_algorithms.train --alg cpo --env SafetyBallCircle-v0
```