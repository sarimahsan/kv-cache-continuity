
### Table: Teacher-Forced Decoupled KV-Cache Ablation at Fixed Depth Schedule ($L = 6$)

$$\begin{array}{lcccccc}
\hline
\textbf{Condition} & \textbf{Attention KV} & \textbf{Readout Layer} & \textbf{PPL} & \Delta\textbf{PPL} & \bar{D}_{\text{KL}} & \textbf{Top-1 Match} \\
\hline
\text{Full Depth Baseline} & \text{True } (K^*, V^*) & h_{L} & 5.7637 & 0.0000 & 0.0000 & 100.0\% \\
\hline
\textbf{Disentanglement Sub-tests:} & & & & & & \\
\text{Readout-Only (Zero KV Corruption)} & \text{True } (K^*, V^*) & h_{6} & 38967281.8733 & +38967276.1096 & 15.8609 & 0.0\% \\
\text{Attention-Only (Direct Copy Floor)} & \text{Direct Copy} & h_{L} & 245873.5880 & +245867.8243 & 10.5297 & 5.0\% \\
\textbf{Attention-Only (Native Proj)} & \mathbf{W_K/W_V + LN} & \mathbf{h_{L}} & \mathbf{43.2165} & \mathbf{+37.4528} & \mathbf{2.1371} & \mathbf{42.9\%} \\
\hline
\textbf{Combined Early-Exit (SkipDecode):} & & & & & & \\
\text{SkipDecode + Direct Copy} & \text{Direct Copy} & h_{6} & 38967281.8733 & +38967276.1096 & 15.8609 & 0.0\% \\
\textbf{SkipDecode + Native Proj} & \mathbf{W_K/W_V + LN} & \mathbf{h_{6}} & \mathbf{38967281.8733} & \mathbf{+38967276.1096} & \mathbf{15.8609} & \mathbf{0.0\%} \\
\hline
\end{array}$$
