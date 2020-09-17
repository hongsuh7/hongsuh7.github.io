---
layout: default
title: Portfolio
---

# Intro
This is the homepage and blog of Hong Suh. Below is my portfolio; other information can be found in the tabs above.

<hr>

# A Generalized Elo System for Tennis Players
## [repo](https://github.com/hongsuh7/tennis-elo), [post 1](https://hongsuh7.github.io/2020/07/07/tennis-1.html), [post 2](https://hongsuh7.github.io/2020/08/13/tennis-2.html), [post 3 (graphics)](https://hongsuh7.github.io/2020/08/26/tennis-3.html)
I built a generalized Elo rating system for tennis players which improves upon the model presented by [FiveThirtyEight](https://fivethirtyeight.com/features/serena-williams-and-the-difference-between-all-time-great-and-greatest-of-all-time/). The improvements are as follows.
1. I eliminate the need to set hyperparameters by hand. This makes my model more generalizable to other settings and less time-consuming to fine-tune.
2. I decreased log-loss error by 1.5% from FiveThirtyEight's model for 2015-2019 test data. 


<p align="center">
	<a href="https://hongsuh7.github.io/2020/08/26/tennis-3.html">
		<img width="500" height="350" src="/assets/tennis-3/big4.png">
	</a>
</p>

<hr>

# Robustness of Neural ODEs
This was a deep learning research project. I conducted statistical tests to determine if decreasing step size in Neural ODE approximations results in increasing robustness against adversarial attacks. I concluded that step size has a significant, but not gigantic, statistical effect on robustness of Neural ODE approximations in certain settings. 

You can read my introduction to Neural ODEs [here](https://hongsuh7.github.io/2020/07/17/neural-ode-intro.html) and a little bit about their adversarial robustness [here](https://hongsuh7.github.io/2020/07/22/neural-ode-robustness.html).

<p align="center">
  <img width="800" height="250" src="/assets/robustness-tests.png">
</p>

<hr>

# Fluctuations of an Exclusion Process
This was a pure math research project. I established quantitative bounds on the distribution of a statistic of an interacting particle system similar to the totally asymmetric simple exclusion process. My visual introduction to exclusion processes can be found [here](https://hongsuh7.github.io/2020/08/13/particles-1.html).

<p align="center">
  <img width="480" height="320" src="/assets/particles-1/tasep_scale.gif">
</p>