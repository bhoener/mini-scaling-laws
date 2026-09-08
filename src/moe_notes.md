# Moe Load Balancing

We want to find the probability that the gate is nonzero for a given expert

$$
P(x, i) = Pr \left((x \cdot W_g)_i + StandardNormal() \cdot Softplus((x \cdot W_{n})_i) \gt kth\_excluding(H(x), k, i) \right)
$$

We know

$$
\Phi(z) = Pr(StandardNormal() \le z)
$$
This means

$$
\Phi \left(kth\_excluding(H(x), k, i) \right) = Pr(StandardNormal() \le kth\_excluding(H(x), k, i))
$$
$$
\Phi \left(\frac{kth\_excluding(H(x), k, i)}{Softplus((x \cdot W_{n})_i)} \right) = Pr(StandardNormal() \le \frac{kth\_excluding(H(x), k, i)}{Softplus((x \cdot W_{n})_i)})
$$
$$
\implies \Phi \left(\frac{kth\_excluding(H(x), k, i)}{Softplus((x \cdot W_{n})_i)} \right) = Pr(StandardNormal() \cdot Softplus((x \cdot W_{n})_i) \le kth\_excluding(H(x), k, i))
$$

since $Softplus(\cdot) \ge 0$

Now,

$$
\Phi \left(\frac{kth\_excluding(H(x), k, i) - (x \cdot W_g)_i}{Softplus((x \cdot W_{n})_i)}  \right) = Pr(StandardNormal() \le \frac{kth\_excluding(H(x), k, i) - (x \cdot W_g)_i}{Softplus((x \cdot W_{n})_i)})
$$

$$
\implies \Phi \left(\frac{kth\_excluding(H(x), k, i) - (x \cdot W_g)_i}{Softplus((x \cdot W_{n})_i)}  \right) = Pr((x \cdot W_g)_i + StandardNormal() \cdot Softplus((x \cdot W_{n})_i) \le kth\_excluding(H(x), k, i))
$$

$$
Pr((x \cdot W_g)_i + StandardNormal() \cdot Softplus((x \cdot W_{n})_i) \gt kth\_excluding(H(x), k, i)) = P(x, i)
$$
$$
= 1 - \Phi \left(\frac{kth\_excluding(H(x), k, i) - (x \cdot W_g)_i}{Softplus((x \cdot W_{n})_i)}\right) = \Phi \left(-( \frac{kth\_excluding(H(x), k, i) - (x \cdot W_g)_i}{Softplus((x \cdot W_{n})_i)} \right)
$$
$$
= \boxed{\Phi \left(\frac{(x \cdot W_g)_i - kth\_excluding(H(x), k, i)}{Softplus((x \cdot W_{n})_i)}\right)}
$$

