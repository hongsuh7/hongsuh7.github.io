---
layout: post
title:  "Collecting Data on Google Colab: Sampling Bias"
excerpt_separator: <!--more-->
---

I've been doing research on approximate Neural ODE models. I set up a system to pick a set of hyperparameters uniformly randomly from a collection of them, evaluate the accuracy of the model under various adversarial attacks, and save the results. I figured this system would get approximately similar numbers of samples from each set of hyperparameters. What I didn't take into account is *sampling bias*. In fact, since I'm running my system on Google Colab, sampling bias ensures that I am not sampling uniformly.

<!--more-->

The thing about Google Colab is that, because it's free (or low-cost with Colab Pro), there is a limit on your runtime. You get kicked out, guaranteed, after 12 hours (24 hour for Colab Pro) on a machine so that you have to reconnect and re-run your program. On average, for one reason or another, I'm cutting my program off after about 6 hours of running. This means that the hyperparameter sets which take longer to train are more likely to be the ones cut off. This phenomenon is called *sampling bias*. I'm not picking a hyperparameter set uniformly; I'm picking a *time* "uniformly," then picking the hyperparameter set corresponding to that time. This subtlety is the reason that my choice is not uniform among all the hyperparameter sets.

Let's examine a simple example, then set clear assumptions for our setting so that we can run simulations to see how far off we are from uniform sampling.

# Simple example: lightbulbs

Suppose I have a room with a single light which is on perpetually. I need this light on at all costs; whenever the lightbulb burns out, I immediately replace it with another. In my stock, I have two types of lightbulbs which are indistinguishable (I spilled them all at an earlier point and put them all in one box). Lightbulb A has an average lifespan of 5 years, and lightbulb B has an average lifespan of 1 second. The standard setting is that both lightbulb lifespans are exponential random variables. 

When a lightbulb burns out, I choose one of the two types of lightbulbs with equal probability. However, if a guest happens to walk into the room at some point in the future, in all probably, the lightbulb in the light will be of type A. 

# Our example

Suppose we have two sets of hyperparameters: A, which runs at 300 seconds, and B, which runs at 1800 seconds. Our units will be in minutes. So let's say ++A_i \sim \text{Exp}(1/5), \quad B_i \sim \text{Exp}(1/30),++ where $A_i,B_i$ are iid and are the time required to run the $i$th training of $A$ and $B$ respectively. There is also the cutting-off process, which, say, is kind of an exponential random variable. Let's say the $T_i$ is the time between the $i$th and $(i+1)$th cutoff time. Our rule is that ++ X_i \sim \text{Exp}(1/360), \quad T_i = \min\\{X_i, 720\\},++ since Google Colab has a maximum runtime of 12 hours (which is 720 minutes). 

Whenever a training completes *or* gets cut off, we begin a new one picking from $\\{A,B\\}$, each with probability 1/2. **As time goes to infinity, what is the proportion of A's among all completed trainings?**

```
n <- 1000

a <- rexp(n = n, rate = 1/6)
b <- rexp(n = n, rate = 1/30)

t <- pmin(rexp(n = n, rate = 1/360), 720)

t <- cumsum(t)
```





