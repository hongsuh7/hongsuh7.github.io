---
layout: post
title:  "Testing Robustness of Neural ODEs against Adversarial Attacks"
excerpt_separator: <!--more-->
---

In my [last post](https://hongsuh7.github.io/2020/07/17/neural-ode-intro.html), I wrote an introduction to Neural ODEs. I made a simple network with one processing layer, one ODE layer, and one fully connected layer. In this post, I will compare the robustness of this model (which we call an "ODENet") to regular ResNets. We see a statistically significant improvement against adversarial attacks when switching from ResNets to ODENets. However, the difference is quite small. This exploration was informed by the paper [On Robustness of Neural Ordinary Differential Equations](https://arxiv.org/pdf/1910.05513.pdf) (Yan et al). 

<!--more-->

## Code 
Code for this blog post is on my [GitHub](https://github.com/hongsuh7).

## Compromises and Opinions

Recall that I said it took six hours for me to train the simple ODENet I described in the previous post. I couldn't do meaningful analyses if one data point takes me half a day to compute. So I decided to approximate the ODENet.

I approximated it in basically the simplest way possible: the regular ResNet is an explicit Euler method with step size 1; I used the explicit Euler method with step size 1/4. **Philosophical tangent:** I think there is actually little reason for anyone to consider implicit methods, adaptive/small step sizes, or higher-order methods for ODENets. Why? 

1. Implicit methods are better than explicit ones when it comes to complicated ODEs with "stiff", or steep, $f$s in $\dot{x} = f(x,t)$. There is no evidence to me that a ResNet block is like this at all. For example, in the original ResNet paper, the blocks basically look like ++ \text{ReLU} \circ \text{conv} \circ \text{ReLU} \circ \text{conv}  ++ and this doesn't seem sufficiently high-gradient enough to justify using implicit methods. In the paper [Invertible Residual Networks](https://arxiv.org/abs/1811.00995), the authors (many of them also wrote the Neural ODEs paper) constrain these blocks to have Lipschitz constant less than one, and this does not affect performance in a significant manner; this suggests to me that even without constraints, these blocks (by themselves) don't have huge Lipschitz constants and therefore doesn't necessitate using implicit methods.

2. As I said, the [Invertible Residual Networks](https://arxiv.org/abs/1811.00995) paper suggests to me that the residual blocks have fairly small Lipschitz constants. So large step sizes with explicit methods should approximate the ODE solution fairly well.

3. And after all, there is no "true" solution we are trying to approximate with discrete steps in the first place. So there is no reason to think that (prior to the experiments) larger error of the numerical method will lead to poorer performance of the classifier. 

I do think that the injectivity of the ODE map helps with something. Yan et al. think that this property of the ODE map is responsible for its adversarial robustness. But we don't need tiny step sizes for this to happen, and I don't think implicit methods really help. For example, if $f$ has Lipschitz constant $L$, then $f/(2L)$ has Lipschitz constant $1/2$, so we need only to do explicit Euler method with step size $1/\lceil 2L \rceil$ in order to get injectivity. If you have opinions on this, please let me know. I'm curious to see what people think.

## Results

I replaced the ODEBlock $\dot{x} = f(x), \~ t\in[0,1]$ in the previous post with the explicit Euler method with step size 1/4. I compared this with the corresponding ResNet, which can be seen as the explicit Euler method with step size 1. I conducted Gaussian noise tests (adding Gaussian noise with the indicated standard deviation to the entire image) and FGSM (fast gradient sign method) tests. I also conducted PGD (projected gradient descent) attacks but they were basically zero all the time for like epsilon = 0.15 so I didn't include them. The summary statistics are below. The following is R code.
```
> nrow(euler1)
[1] 15

> nrow(euler4)
[1] 11

> sapply(X = euler1, FUN = mean)
    plain   gaus0.5   gaus1.0   gaus1.5  fgsm0.15   fgsm0.3   fgsm0.5 
98.146667 93.480667 67.374000 41.986667 57.528000 16.608000  1.681333 

> sapply(X = euler1, FUN = sd)
     plain    gaus0.5    gaus1.0    gaus1.5   fgsm0.15    fgsm0.3    fgsm0.5 
 0.1836404  2.1582515  5.5016579  4.4575805 27.3154483 13.2658871  2.8893545 

> sapply(X = euler4, FUN = mean)
    plain   gaus0.5   gaus1.0   gaus1.5  fgsm0.15   fgsm0.3   fgsm0.5 
98.331818 93.660000 65.752727 40.272727 63.420000 18.394545  1.952727

> sapply(X = euler4, FUN = sd)
     plain    gaus0.5    gaus1.0    gaus1.5   fgsm0.15    fgsm0.3    fgsm0.5 
 0.1373913  2.3716281  5.2053032  3.0574568 23.4010983 18.3954023  3.9513772 
```

Note the standard deviations! What the summary statistics don't tell you is that the distribution of numbers for the fgsm tests were fairly bimodal; more than half the time, these models do very poorly on the fgsm tests (like <5% on fgsm0.3) and the other times, they did a lot better (like >20% on fgsm0.5). You can take a look at the dataset on my [GitHub](https://github.com/hongsuh7). Carrying out a statistical test is almost meaningless here, because obviously there is no statisically significant difference between the two methods.

I still don't know why the distribution is bimodal on adversarial tests. I'd love to find out---please let me know if you know why this happens.

## Adversarial Training: Results

To really see the potential for robustness in these models, we should do some adversarial training before testing. To do this, with each mini-batch I made my usual gradient descent step, then I also calculated the corresponding fgsm attacks with epsilon=0.15. I made another gradient descent step with these perturbed images and their corresponding labels, but the loss function I multiplied by 0.5 (to tell the model that the original images are more important than the perturbed ones, but not by much). 

```
> nrow(euler1_adv)
[1] 24

> nrow(euler4_adv)
[1] 14

> sapply(euler1_adv, mean)
   plain  gaus0.5  gaus1.0  gaus1.5 fgsm0.15  fgsm0.3  fgsm0.5  fgsm0.7 
98.44625 97.19875 75.43000 44.59667 95.14208 86.97542 60.76250 26.09750 

> sapply(euler1_adv, sd)
    plain   gaus0.5   gaus1.0   gaus1.5  fgsm0.15   fgsm0.3   fgsm0.5   fgsm0.7 
0.1407376 0.4559111 4.5464013 4.7227900 0.6125001 1.8283872 6.3299538 8.5499999 

> sapply(euler4_adv, mean)
   plain  gaus0.5  gaus1.0  gaus1.5 fgsm0.15  fgsm0.3  fgsm0.5  fgsm0.7 
98.61857 97.49929 74.84214 44.54429 95.72857 88.15000 59.82071 26.48643 

> sapply(euler4_adv, sd)
    plain   gaus0.5   gaus1.0   gaus1.5  fgsm0.15   fgsm0.3   fgsm0.5   fgsm0.7 
0.1029456 0.6656687 4.3496387 3.5756306 0.3303811 1.6361399 6.7798326 7.5182287 
```

Ok, now we're cooking. The variances are down -- in fact they are very small for fgsm0.15 and fgsm0.3. We can do a regular t-test with the fgsm data.

```
> t.test(euler1_adv$fgsm0.15, euler4_adv$fgsm0.15)

	Welch Two Sample t-test

data:  euler1_adv$fgsm0.15 and euler4_adv$fgsm0.15
t = -3.8317, df = 35.875, p-value = 0.000493
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -0.8969503 -0.2760259
sample estimates:
mean of x mean of y 
 95.14208  95.72857 

> t.test(euler1_adv$fgsm0.3, euler4_adv$fgsm0.3)

	Welch Two Sample t-test

data:  euler1_adv$fgsm0.3 and euler4_adv$fgsm0.3
t = -2.0431, df = 29.877, p-value = 0.04995
alternative hypothesis: true difference in means is not equal to 0
95 percent confidence interval:
 -2.3488747386 -0.0002919281
sample estimates:
mean of x mean of y 
 86.97542  88.15000 
```

The 4-step Euler network is statistically significantly better against adversarial attacks than the 1-step Euler network! But, as you can see, it's not *that* much better. The other two attacks, fgsm0.5 and fgsm0.7, have no statistically significant difference.

We can also do t-tests on the other columns. 4-step Euler is definitely better than 1-step Euler for the plain (no perturbation) tests, whether adversarially trained or not.

Note that the adversarially trained networks are better than the regularly trained networks even with no attacks, and with Gaussian noise attacks.

## Conclusion

We see that even with this simple structure, we see a statistically significant improvement in the 4-step Euler network from the 1-step Euler network (regular ResNet). It does take more computational cost to backprop through more functions, though the number of parameters is equal. But yes, we do conclude that ODENets are more robust against adversarial attacks, and a little better in general, than regular ResNets (but not by much).
