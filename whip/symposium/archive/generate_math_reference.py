"""Generate mathematical reference sheet for BeadMPGNN forward pass."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14, 'mathtext.fontset': 'cm'})

fig, ax = plt.subplots(1, 1, figsize=(14, 20))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 28)

y = 27.5
lx = 0.3  # left margin

def title(text, y_pos):
    ax.text(lx, y_pos, text, fontsize=18, fontweight="bold", color="#1565C0")
    ax.plot([lx, 9.7], [y_pos - 0.25, y_pos - 0.25], color="#1565C0", lw=1.5)
    return y_pos - 0.7

def line(text, y_pos, fontsize=14, color="black", indent=0):
    ax.text(lx + indent, y_pos, text, fontsize=fontsize, color=color)
    return y_pos - 0.55

def math(text, y_pos, indent=0.3):
    ax.text(lx + indent, y_pos, text, fontsize=15, color="#333333",
            fontfamily="serif")
    return y_pos - 0.6

def gap(y_pos, size=0.3):
    return y_pos - size

# ============================================================
y = title("BeadMPGNN — Full Forward Pass (K=1)", y)
y = line("For one bead $i$ with left neighbor $L$ and right neighbor $R$", y, fontsize=13, color="gray")
y = gap(y)

# ============================================================
y = title("1. Inputs", y)
y = math(r"$\mathbf{x}_{node} = [v_x,\ v_y,\ m,\ \mathrm{is\_fixed}] \in \mathbb{R}^{4}$", y)
y = math(r"$\mathbf{x}_{edge}^L = [\Delta x,\ \Delta y,\ \Delta v_x,\ \Delta v_y,\ m_L,\ \ell_0] \in \mathbb{R}^{6}$", y)
y = math(r"$\mathbf{x}_{edge}^R = [\Delta x,\ \Delta y,\ \Delta v_x,\ \Delta v_y,\ m_R,\ \ell_0] \in \mathbb{R}^{6}$", y)
y = line("where $\\Delta$ quantities are relative to the neighbor", y, fontsize=12, color="gray", indent=0.3)
y = gap(y)

# ============================================================
y = title("2. Input Normalization", y)
y = math(r"$\hat{\mathbf{x}}_{node} = (\mathbf{x}_{node} - \boldsymbol{\mu}_{node}) \,/\, \boldsymbol{\sigma}_{node}$", y)
y = math(r"$\hat{\mathbf{x}}_{edge} = (\mathbf{x}_{edge} - \boldsymbol{\mu}_{edge}) \,/\, \boldsymbol{\sigma}_{edge}$", y)
y = line("$\\boldsymbol{\\mu}$, $\\boldsymbol{\\sigma}$ computed once from training set, fixed during training", y, fontsize=12, color="gray", indent=0.3)
y = gap(y)

# ============================================================
y = title("3. ResMLPBlock (used by all encoders)", y)
y = line("Given input $\\mathbf{z}$ with input dim $d_{in}$, hidden dim $H = 64$:", y, fontsize=13, color="gray")
y = gap(y, 0.15)
y = math(r"$\mathbf{z}' = W_{proj}\,\mathbf{z} + \mathbf{b}_{proj}$                    $W_{proj} \in \mathbb{R}^{H \times d_{in}}$", y)
y = math(r"$\hat{\mathbf{z}} = \mathrm{LayerNorm}(\mathbf{z}')$", y)
y = math(r"$\mathbf{a} = \mathrm{GELU}\!\left(W_1\,\hat{\mathbf{z}} + \mathbf{b}_1\right)$          $W_1 \in \mathbb{R}^{2H \times H}$    (expand)", y)
y = math(r"$\mathbf{o} = W_2\,\mathbf{a} + \mathbf{b}_2$                        $W_2 \in \mathbb{R}^{H \times 2H}$    (contract)", y)
y = math(r"$\mathrm{output} = \mathbf{z}' + \mathbf{o}$                          (residual connection)", y)
y = gap(y)

# ============================================================
y = title("4. Node Encoder", y)
y = math(r"$\mathbf{h}_{self} = \mathrm{ResMLPBlock}(\hat{\mathbf{x}}_{node})$           $\mathbb{R}^{4} \to \mathbb{R}^{64}$", y)
y = gap(y)

# ============================================================
y = title("5. Edge Encoder (shared weights for L and R)", y)
y = math(r"$\mathbf{h}_{edge}^L = \mathrm{ResMLPBlock}(\hat{\mathbf{x}}_{edge}^L)$        $\mathbb{R}^{6} \to \mathbb{R}^{64}$", y)
y = math(r"$\mathbf{h}_{edge}^R = \mathrm{ResMLPBlock}(\hat{\mathbf{x}}_{edge}^R)$        $\mathbb{R}^{6} \to \mathbb{R}^{64}$      (same weights)", y)
y = gap(y)

# ============================================================
y = title("6. Message MLP", y)
y = math(r"$\mathbf{m}_L = \mathrm{ResMLPBlock}(\mathbf{h}_{edge}^L)$              $\mathbb{R}^{64} \to \mathbb{R}^{64}$", y)
y = math(r"$\mathbf{m}_R = \mathrm{ResMLPBlock}(\mathbf{h}_{edge}^R)$              $\mathbb{R}^{64} \to \mathbb{R}^{64}$      (same weights)", y)
y = gap(y)

# ============================================================
y = title("7. Ghost Masking", y)
y = math(r"$\mathbf{m}_L = \mathbb{1}_{has\_left} \cdot \mathbf{m}_L$                (zero if no left neighbor)", y)
y = math(r"$\mathbf{m}_R = \mathbb{1}_{has\_right} \cdot \mathbf{m}_R$              (zero if no right neighbor)", y)
y = gap(y)

# ============================================================
y = title("8. Node Update MLP", y)
y = math(r"$\mathbf{c} = [\mathbf{h}_{self}\ \|\ \mathbf{m}_L\ \|\ \mathbf{m}_R] \in \mathbb{R}^{192}$       (concatenation)", y)
y = math(r"$\mathbf{h}_{updated} = \mathrm{ResMLPBlock}(\mathbf{c})$            $\mathbb{R}^{192} \to \mathbb{R}^{64}$", y)
y = gap(y)

# ============================================================
y = title("9. Output Head", y)
y = math(r"$\hat{\mathbf{y}} = W_{out}\,\mathrm{LayerNorm}(\mathbf{h}_{updated}) + \mathbf{b}_{out}$     $W_{out} \in \mathbb{R}^{4 \times 64}$", y)
y = gap(y)

# ============================================================
y = title("10. Output Denormalization", y)
y = math(r"$\mathbf{y} = \hat{\mathbf{y}} \cdot \boldsymbol{\sigma}_y + \boldsymbol{\mu}_y$", y)
y = math(r"$\mathbf{y} = [\Delta x,\ \Delta y,\ \Delta v_x,\ \Delta v_y] \in \mathbb{R}^{4}$", y)
y = gap(y)

# ============================================================
y = title("11. State Update", y)
y = math(r"$\mathbf{pos}_{new} = \mathbf{pos} + [\Delta x,\ \Delta y]$", y)
y = math(r"$\mathbf{vel}_{new} = \mathbf{vel} + [\Delta v_x,\ \Delta v_y]$", y)
y = gap(y)

# Footer
ax.text(5, y - 0.3, "Total: 80,324 trainable parameters    |    Hidden dim $H = 64$    |    Activation: GELU",
        ha="center", fontsize=13, color="gray", style="italic")

plt.tight_layout()
plt.savefig("mpgnn_math_reference.png", dpi=150, bbox_inches="tight")
print("Saved mpgnn_math_reference.png")
