---
layout: default
title: Portfolio
---

# A Generalized Elo System for Tennis Players
I built a generalized Elo rating system for tennis players which improves upon the model presented by [FiveThirtyEight](https://fivethirtyeight.com/features/serena-williams-and-the-difference-between-all-time-great-and-greatest-of-all-time/). The improvements are as follows.
1. I eliminate the need to set hyperparameters by hand. This makes my model more generalizable to other settings and less time-consuming to fine-tune.
2. I incorporate surface in a novel and natural way. 
I improved log-loss error by 1.5% from FiveThirtyEight's model for 2015-2019 test data.

My Github repo is [here](https://github.com/hongsuh7/tennis-elo); my corresponding [first post](https://hongsuh7.github.io/2020/07/07/tennis-1.html) and [second post](https://hongsuh7.github.io/2020/08/13/tennis-2.html). Below are some sample ratings as of end of 2019.

<p align="center">
  <img width="400" height="180" src="/assets/tennis-2/ratings.png">
</p>

# Robustness of Neural ODEs (in progress)
This is a deep learning research project. I conducted statistical tests to determine if decreasing step size in Neural ODE approximations results in increasing robustness against adversarial attacks. I concluded that step size has a significant statistical effect on robustness of Neural ODE approximations in certain settings. 

To further study this relationship, I set up an automated system which, at each iteration, chooses between 48 hyperparameter configurations, trains the model, then tests robustness. I am in the process of comparing the effects of certain hyperparameter changes to adversarial robustness. You can read my introduction to Neural ODEs [here](https://hongsuh7.github.io/2020/07/17/neural-ode-intro.html) and a little bit about their adversarial robustness [here](https://hongsuh7.github.io/2020/07/22/neural-ode-robustness.html).

<p align="center">
  <img width="800" height="250" src="/assets/robustness-tests.png">
</p>

# Fluctuations of an Exclusion Process
This was a pure math research project. I established quantitative bounds on the distribution of a statistic of an interacting particle system similar to the totally asymmetric simple exclusion process. A visual introduction can be found [here](https://hongsuh7.github.io/2020/08/13/particles-1.html).

<p align="center">
  <img width="480" height="320" src="/assets/particles-1/tasep_scale.gif">
</p>