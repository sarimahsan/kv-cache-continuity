
### Table: Teacher-Forced Decoupled KV-Cache Ablation at Fixed Depth Schedule ($L = 8$)

$$\begin{array}{lcccccc}
\hline
\textbf{Condition} & \textbf{Attention KV} & \textbf{Readout Layer} & \textbf{PPL} & \Delta\textbf{PPL} & \bar{D}_{\text{KL}} & \textbf{Top-1 Match} \\
\hline
\text{Full Depth Baseline} & \text{True } (K^*, V^*) & h_{L} & 8.6876 & 0.0000 & 0.0000 & 100.0\% \\
\hline
\textbf{Disentanglement Sub-tests:} & & & & & & \\
\text{Readout-Only (Zero KV Corruption)} & \text{True } (K^*, V^*) & h_{8} & 147615.4519 & +147606.7644 & 9.8892 & 2.1\% \\
\text{Attention-Only (Direct Copy Floor)} & \text{Direct Copy} & h_{L} & 1025.7062 & +1017.0186 & 4.6518 & 23.5\% \\
\textbf{Attention-Only (Native Proj)} & \mathbf{W_K/W_V + LN} & \mathbf{h_{L}} & \mathbf{17.0351} & \mathbf{+8.3476} & \mathbf{0.6081} & \mathbf{69.6\%} \\
\hline
\textbf{Combined Early-Exit (SkipDecode):} & & & & & & \\
\text{SkipDecode + Direct Copy} & \text{Direct Copy} & h_{8} & 147615.4519 & +147606.7644 & 9.8892 & 2.1\% \\
\textbf{SkipDecode + Native Proj} & \mathbf{W_K/W_V + LN} & \mathbf{h_{8}} & \mathbf{147615.4519} & \mathbf{+147606.7644} & \mathbf{9.8892} & \mathbf{2.1\%} \\
\hline
\end{array}$$
