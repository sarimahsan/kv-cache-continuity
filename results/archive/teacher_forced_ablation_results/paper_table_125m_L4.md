
### Table: Teacher-Forced Decoupled KV-Cache Ablation at Fixed Depth Schedule ($L = 4$)

$$\begin{array}{lcccccc}
\hline
\textbf{Condition} & \textbf{Attention KV} & \textbf{Readout Layer} & \textbf{PPL} & \Delta\textbf{PPL} & \bar{D}_{\text{KL}} & \textbf{Top-1 Match} \\
\hline
\text{Full Depth Baseline} & \text{True } (K^*, V^*) & h_{L} & 8053.2866 & 0.0000 & 0.0000 & 100.0\% \\
\hline
\textbf{Disentanglement Sub-tests:} & & & & & & \\
\text{Readout-Only (Zero KV Corruption)} & \text{True } (K^*, V^*) & h_{4} & 5973196.4831 & +5965143.1965 & 4.5069 & 49.7\% \\
\text{Attention-Only (Direct Copy Floor)} & \text{Direct Copy} & h_{L} & 20417.7097 & +12364.4231 & 0.7704 & 63.0\% \\
\textbf{Attention-Only (Native Proj)} & \mathbf{W_K/W_V + LN} & \mathbf{h_{L}} & \mathbf{8245.6495} & \mathbf{+192.3629} & \mathbf{0.3020} & \mathbf{65.6\%} \\
\hline
\textbf{Combined Early-Exit (SkipDecode):} & & & & & & \\
\text{SkipDecode + Direct Copy} & \text{Direct Copy} & h_{4} & 5973196.4831 & +5965143.1965 & 4.5069 & 49.7\% \\
\textbf{SkipDecode + Native Proj} & \mathbf{W_K/W_V + LN} & \mathbf{h_{4}} & \mathbf{5973196.4831} & \mathbf{+5965143.1965} & \mathbf{4.5069} & \mathbf{49.7\%} \\
\hline
\end{array}$$
