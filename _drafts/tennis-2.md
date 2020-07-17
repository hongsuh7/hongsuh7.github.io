---
layout: post
title:  "A Generalized Elo System for Tennis Players, part 2"
excerpt_separator: <!--more-->
---


In this post, we will take our previous Elo system for tennis players and add playing surface as a parameter. There are a few ways in which surface has been taken into account.

<!--more-->

1. (Surface-only) treat each surface as a different sport altogether, so that each player has three ratings that don't interact with one another.
2. (Weighted average) take the surface-specific ratings in item 1 above and the all-surfaces ratings developed in our previous post, then take a weighted average of them, minimizing the log-loss error.
3. (Surface-dependent K-factor) According to the surface being played on, update each player's surface-specific rating according to a different K-factor and take the win probability from the corresponding surface-specific ratings.

The first and second are implemented by Jeff Sackmann, the tennis data god, where the weighted average is the actual average. The third is the idea introduced in this post, which seems fairly natural to me and perhaps a little less ad-hoc than taking the average between surface-only and surface-agnostic ratings. So let's explain how the surface-dependent K-factor (SDKF) model works.

## SDKF model

Define ++\sigma(x) = \exp(x) / (\exp(x) + 1),++ the logistic function. If player one (p1) has rating $x$ and player two (p2) has rating $y$, the probability that p1 wins is given by $\sigma(x-y)$. Suppose $w=1$ if p1 wins and $w=0$ if p1 loses. After the match, the ratings are updated with the rule ++x \mapsto x + (-1)^{w+1} K(n_1)\sigma((-1)^w(x-y)),\quad y \mapsto y+(-1)^w K(n_2)\sigma((-1)^w (x-y)),++ where $K$ is a function of the number of matches played by p1 ($n_1$) and the number of matches played by p2 ($n_2$). The function $K$ is of the form ++K(n) = \frac{a}{(b + n)^c}.++

To define surface-specific ratings, we can do the following. Let $A$ be a $3\times 3$ matrix. We map surfaces to indices: index 1 refers to clay, 2 to grass, 3 to hard. Now let $\vec{x},\vec{y}\in \mathbb{R}^3$ be the ratings of p1 and p2, respectively. Specifically, ++\vec{x} = (x_1,x_2,x_3)++ and $x_1$ is the p1 clay rating, $x_2$ is the p1 grass rating, and so on. Define $\sigma(\vec{x}) = (\sigma(x_1),\sigma(x_2),\sigma(x_3))$. If $a_{ij}$ is the $(i,j)$ entry of $A$, then we make the following change to the update rule: ++\vec{x} \mapsto \vec{x} + (-1)^{w+1}K(n_1)A\sigma((-1)^w(\vec{x}-\vec{y})), \quad \vec{y} \mapsto \vec{y} + (-1)^w K(n_2)A\sigma((-1)^w(\vec{x}-\vec{y})).++

The matrix $A$ consists of the speed with which to update each of the three ratings, given the surface being played on. For example, if the match is being played on grass, we intuit that the result shouldn't have a large effect on the players' clay rating, but it should have a large effect on the players' grass rating. On the other hand, if the match is being played on hard, we might think that it should have an equal effect on the players' grass and clay ratings.

Finally, let's determine the win probability and the interpretation of the matrix $A$. If ++\vec{s}=\begin{cases} \vec{e}_1 &\quad \text{ if clay} \\\ \vec{e}_2 &\quad \text{ if grass} \\\ \vec{e}_3 &\quad \text{ if hard} \end{cases}++ is the vector denoting surface being played on, then the win probability of p1 is ++\sigma(\vec{x}-\vec{y})\cdot \vec{s}.++ This indicates that **$a_{ij}$ is the update speed for the players' surface $i$ rating if the playing surface is $j$**.

### Special cases
It is instructive to examine special cases of $A$.
1. If $A$ is the identity matrix, then no surface affects any other surface, and all the update coefficients are equal. So this would be equivalent to treating each surface as a different sport altogether (Surface-only ratings).
2. If $A$ is the all-ones matrix, then all surfaces are treated equally. This results in surface-agnostic ratings, which is the classical setting.

Based on these two extremes, we expect an effective $A$ to have heavy diagonal entries but nonzero off-diagonal entries, all positive. For our $K$, we take $a,b,c$ to be ++(a,b,c) = (0.47891635, 4.0213623 , 0.25232273)++ based on training data from 1980 to 2014, from the previous post. Then we initialize the entries of $A$ to be uniform random numbers between 0 and 1.5.
