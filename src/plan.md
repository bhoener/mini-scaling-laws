Seems like the hard part about MoE is the parallelism. I will first try to implement it for a single device, then work on ddp.

For single device:
 - Take hidden state $x$ -> gating layer $W_G$
 - Sample random noise $\epsilon \sim \mathcal{N}(0, I_d)$
 - Take hidden state -> noise layer $W_N$
 - Multiply $\epsilon \cdot xW_N$ -> softplus
 - add to gates $xW_G + \epsilon \cdot \text{softplus}(xW_N)$
 - Keep top k, set rest to -inf -> softmax