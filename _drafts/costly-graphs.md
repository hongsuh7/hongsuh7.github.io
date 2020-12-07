---
layout: post
title:  "Costly Graphs Solution"
excerpt_separator: <!--more-->
---

I came across a HackerRank problem that seemed quite mathematically involved, though it was labeled a medium. The title of the problem is [Costly Graphs](https://www.hackerrank.com/challenges/costly-graphs/problem). 

<!--more-->

# The problem
Let $G$ be a simple undirected graph. Given a number $K>0$, the cost of the graph $C^K(G)$ is the quantity ++ C^K(G) = \sum_{v \in G} d(v)^K, ++ where $d(v)$ is the degree of vertex $v$. The question is as follows. Suppose $(N_1,K_1),\ldots, (N_T, K_T)$ are $T$ pairs of positive integers. For each pair, calculate ++ \sum_{|G| = N} C^K(G) \pmod{1005060097}. ++ We do not collapse isomorphic graphs.

The constraints are ++ \begin{align}1 \leq T &\leq 2 \cdot 10^5,\\\ \quad 1 \leq N &\leq 10^9, \\\ \quad 1 \leq K &\leq 2\cdot 10^5,  \\\ \quad \sum_{i=1}^T K_i &\leq 2 \cdot 10^5. \end{align} ++

## Example
For example, for $N=3$ and $K=2$, there are 8 possible graphs since there are 3 pairs of vertices and each of them can either have an edge or not. There is one with no edges, which doesn't contribute to the cost. There are three with one edge, which contribute $3(1^2+1^2)=6$. There are three with two edges, which contribute $3(2^2+1^2+1^2)=18$. There is one with three edges, which contributes $2^2+2^2+2^2 = 12$. So for the pair $(N,K)=(3,2)$, we'd return ++(6+18+12) \pmod{1005060097}=36.++

# Solution

## Setup

Note that what we are trying to compute is a double sum. And what do we do if we have a double sum? Switch the sums. Below we label the vertices $1,2,\ldots,N$. 
++ \begin{align} C_N^K &:= \sum_{|G|=N} \sum_{v \in G} d(v)^k \\\
    &= \sum_{v=1}^N \sum_{G \ni v, ~ |G|=N} d(v)^k \\\
    &= N \sum_{|G|=N} d(1)^k \quad \text{(by symmetry)} \\\
    &= N \sum_{i=0}^{N-1} |\\{G ~|~ d(1) = i\\}| ~ i^k. \end{align} ++

To find the coefficient of $i^k$, we just need to count the number of graphs which have $i$ edges coming out of vertex $1$. There are $N-1$ potential edges emanating from node $1$, and $i$ of them must be edges and $N-1-i$ of them must be empty. There are ${N-1 \choose i}$ ways to choose these nonempty edges. Then the rest of the $N-1$ nodes can be whatever graph they want. So this contributes a factor of $2^{N-1 \choose 2}$. Therefore 
++ |\\{ G | d(1) = i\\}| = {N-1 \choose i} 2^{(N-1)(N-2)/2}. ++

Our final formula is ++ \boxed{C_N^K = N 2^{(N-1)(N-2)/2} \sum_{i=0}^{N-1} {N-1 \choose i} i^k.} ++ 

I wrote this code (knowing it wasn't quite fast enough yet, since $N$ can go up to $10^9$) to check if my formla is correct. Turns out it is, but of course it times out since it's $O(NT)$.

```
import math

prime = 1005060097

def choose(n,k):
    return math.factorial(n) // math.factorial(n-k) // math.factorial(k)

def cost(n,k):
    if n <= 1:
        return 0
    coef = n * pow(2, (n-1) * (n-2) // 2, prime)
    return (coef * sum(choose(n-1, i) * pow(i,k,prime) for i in range(n))) % prime

T = int(input())

for i in range(T):
    n,k = [int(i) for i in input().split()]
    print(cost(n,k))
```

But I do like the simplicity. Now we must make it more complicated to make it faster. There is definitely a complicated formula for that sum. I empirically verified the following identities using OEIS:
++ \begin{align} \sum_{i=0}^{m} {m \choose i} i^2 &= m(m+1)2^{m-2}, \\\ \sum_{i=0}^m {m\choose i} i^3 &= m^2(m+3) 2^{m-3}, \\\ \sum_{i=0}^m {m \choose i} i^4 &= m(m+1)(m^2+5m-2) 2^{m-4}. \end{align} ++ 

Let's look toward a closed form solution for $C_N^K$. Let ++S_m^k := \sum_{i=0}^m {m \choose i} i^k. ++
Using the recurrence relation for ${m\choose i}$, we get
++ 
\begin{align}
    S_m^k &=\sum_{i=0}^m {m \choose i} i^k \\\
    &= \sum_{i=0}^m \left( {m-1 \choose i} + {m-1 \choose i-1} \right) i^k \\\
    &= S_{m-1}^k + \sum_{i=0}^{m-1} {m-1 \choose i} (i+1)^k \\\
    &= S_{m-1}^k + \sum_{i=0}^{m-1} {m-1 \choose i} \sum_{\ell=0}^k {k \choose \ell} i^\ell \\\
    &= S_{m-1}^k + \sum_{\ell=0}^k {k \choose \ell} \sum_{i=0}^{m-1} {m-1 \choose i} i^\ell \\\
    &=\boxed{ S_{m-1}^k + \sum_{\ell=0}^k {k \choose \ell} S_{m-1}^\ell.}
\end{align}
++
Whew, ok. I tried for a long time to find a closed form for this recurrence relation, and it's doable for fixed $k$. But I can't get a formula given $m$ and $k$. This is a problem because right now, my algorithm is $O(NT)$ which is too large. But there is a painful way to compute the closed form for the recurrence relation by incrementing $k$. That would make my algorithm $O(K+T)$ (sorry I've been switching $k$ and $K$) by precomputing the formulas then running the test cases, which would be way better.

## Switching Formulas: Proof

To find the closed form with the above recurrence relation, I think I'd need to solve a $k$-dimensional linear system incrementally, which seems too complicated if there is no obvious pattern (which there is not). Instead, we switch to another recurrence.

Consider $f_m(x) = (1+x)^m$. Then let $(Tf_m)(x) = x f'(x)$. The claim is that ++ (T^kf_m)(1) = S_m^k. ++ Here is the proof. If $p(x)$ is a polynomial in $x$, then
++\begin{align}
    p(x) &= a_0 + a_1x + a_2x^2 + a_3x^3 + \cdots \\\
    Tp(x) &= a_1x + 2a_2x^2 + 3a_3x^3 + \cdots \\\
    T^2p(x) &= a_1x + 2^2 a_2x^2 + 3^2a_3x^3 + \cdots \\\
    T^kp(x) &= a_1x + 2^k a_2x^2 + 3^k a_3x^3 + \cdots.
\end{align}++

Start with $T^0f_m(x) = (1+x)^m = \sum_{i=0}^m {m \choose i} x^i$. We simply compute
++ T^kf_m(x) = \sum_{i=0}^m {m\choose i} i^k x^i. ++ Plugging in $x=1$ completes the proof.

## Computing Alternate Formula

Now we work with this $T^kf_m(x)$ formula. 

To compute $(T^kf_m)(1)$ on the computer, we need to discover a little more structure in $T^kf_m(x)$. We claim that $T^kf_m(x)$ is of the form ++ \boxed{T^kf_m(x) = p_k(x,m) (1+x)^{m-k}} \quad \text{ for } 0\leq k \leq m ++ where $p_k(x,m)$ is a bivariate polynomial of degree $k$ in both $x$ and $m$ (e.g. $p_2(x,m) = mx(mx+1)$). Certainly this is true for $k=0$. If it's true for $k=1$, then 
++ \begin{align}
    T^kf_m(x) &= T(T^{k-1}f_m)(x) \\\
    &= T(p_{k-1}(x,m) (1+x)^{m-k+1}) \\\
    &= x\left(p_{k-1}'(x,m) (1+x)^{m-k+1} + p_{k-1}(x,m) (m-k+1)(1+x)^{m-k}\right) \\\
    &= x((1+x)\cdot p_{k-1}'(x,m) + (m-k+1)\cdot p_{k-1}(x,m)) (1+x)^{m-k}\\\
    &=: p_k(x,m) (1+x)^{m-k}.
\end{align} ++
The coefficient of $(1+x)^{m-k}$ above is $p_k(x,m)$, and we see that it has degree $k$ in both $x$ and $m$. 

As we see above, the expression $p_k(x,m)$ follows the rule ++ \boxed{p_k(x,m) = x(1+x)\cdot p_{k-1}'(x,m) + x(m-k+1)\cdot p_{k-1}(x,m), \quad p_0(x,m) = 1.} ++ 

Also a note that for $k> m$ we see that $T^kf_m(x) = T^{k-m} p_m(x,m)$. But this computation should be easier than the case $k\leq m$. It is simply a repeated shift and multiplication.

We must now think about how to store the formula once we get it. For each $k$, we will have a hash table to store the coefficient of $x^im^j$. Theoretically this could take $O(k^2)$ space and time, but in practice it seems to be okay.

```
formula_cache = []

def grad(formula):
    ''' computes the gradient of a polynomial given its formula. '''
    gradient = {}
    for (i,j) in formula:
        if i > 0:
            gradient[(i-1,j)] = gradient.get((i-1,j), 0) + i * formula[(i,j)]
    return gradient

def build_cache(k):
    ''' computes p_k(x,m) up to k. '''

    for ell in range(k+1):
        if ell == 0: 
            formula_cache.append({(0,0): 1})
        else:
            prev_formula = formula_cache[ell-1]
            curr_formula = {}
            grad_prev = grad(prev_formula)

            for (i,j) in prev_formula:
                term = prev_formula[(i,j)]
                curr_formula[(i+1,j+1)] = curr_formula.get((i+1,j+1), 0) + term
                curr_formula[(i+1,j)]   = curr_formula.get((i+1,j), 0)   + (-ell + 1) * term
            
            for (i,j) in grad_prev:
                term = grad_prev[(i,j)]
                curr_formula[(i+2,j)] = curr_formula.get((i+2,j), 0) + term
                curr_formula[(i+1,j)] = curr_formula.get((i+1,j), 0) + term

            formula_cache.append(curr_formula)
```

Now for each test case, we simply look up the largest $K$, build our cache up to that $K$, then compute. The final code looks like this.
```

```




