
### Table: Teacher-Forced Decoupled KV-Cache Ablation at Fixed Depth Schedule ($L = 18$)

$$\begin{array}{lcccccc}
\hline
\textbf{Condition} & \textbf{Attention KV} & \textbf{Readout Layer} & \textbf{PPL} & \Delta\textbf{PPL} & \bar{D}_{\text{KL}} & \textbf{Top-1 Match} \\
\hline
\text{Full Depth Baseline} & \text{True } (K^*, V^*) & h_{L} & 5.7637 & 0.0000 & 0.0000 & 100.0\% \\
\hline
\textbf{Disentanglement Sub-tests:} & & & & & & \\
\text{Readout-Only (Zero KV Corruption)} & \text{True } (K^*, V^*) & h_{18} & 715406.4805 & +715400.7168 & 11.7805 & 1.3\% \\
\text{Attention-Only (Direct Copy Floor)} & \text{Direct Copy} & h_{L} & 112.7495 & +106.9858 & 3.2291 & 33.9\% \\
\textbf{Attention-Only (Native Proj)} & \mathbf{W_K/W_V + LN} & \mathbf{h_{L}} & \mathbf{9.8168} & \mathbf{+4.0531} & \mathbf{0.7335} & \mathbf{70.9\%} \\
\hline
\textbf{Combined Early-Exit (SkipDecode):} & & & & & & \\
\text{SkipDecode + Direct Copy} & \text{Direct Copy} & h_{18} & 715406.4805 & +715400.7168 & 11.7805 & 1.3\% \\
\textbf{SkipDecode + Native Proj} & \mathbf{W_K/W_V + LN} & \mathbf{h_{18}} & \mathbf{715406.4805} & \mathbf{+715400.7168} & \mathbf{11.7805} & \mathbf{1.3\%} \\
\hline
\end{array}$$
