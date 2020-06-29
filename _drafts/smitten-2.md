---
layout: post
title:  "Is Smitten Ice Cream Really That Creamy?: A Mathematical Perspective, part 2"
excerpt_separator: <!--more-->
---

## Probability Distributions

The phrase ``a probability distribution on the set of particle configurations`` is a mouthful, but the idea is simple: assign a probability to each particle configuration. For example,

<!-- insert here -->

There are many different probability distributions we can put on any $B_n$. Most are not physically meaningful. For example, I could define a probability distribution on $B_2$ as follows.

<!-- insert here -->

Sure, this is a probability distribution. But it is not very useful because it has no connection to physical reality. This is where *energy* comes in. 

<!-- insert here -->

It is useful to think about the extremes of $\beta$. If $\beta=0$, then every particle configuration is equally likely, no matter their energies. If $\beta=\infty$, then the lowest-energy particle configuration has probability one while the others have probability zero. (If there are many lowest-energy configurations, they all have the same probability.) The parameter $\beta$ has an important physical meaning which we will describe now.

## Inverse Temperature
The parameter $\beta$ is often called *inverse temperature*. 