"""v14 text-track input/output heads.

Public symbols:
  - TextCodebookHead: trainable projections + small embeddings for the text track.
  - tokenize_with_eos_fill: tokenize texts and EOS-fill positions after natural EOS.
"""
import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List


class TextCodebookHead(nn.Module):
    """v14 text-track input/output heads.

    Trainable parameters owned here:
      - text_token_projs: ModuleDict -- per-active-layer Linear(D1 -> D1 or D2)
        keyed "layer_{i}", mirroring add_token_projs (moma_v11_final_super.py:488-494).
      - text_head: Linear(D1 -> 768) -- single OUTPUT projection (a-layer output width).
      - text_pos_emb: Embedding(L_text, D1) -- learned 1D pos emb, zero-init.
      - role_emb_d1: Embedding(4, cond_d1_eff) -- zero-init.
      - role_emb_d2: Embedding(4, cond_d2_eff) -- zero-init.

    Role ids: {0: register, 1: ctx_motion, 2: x_motion, 3: lang}.

    The T5 codebook (frozen 768-dim embeddings) is NOT owned here; it lives as a
    buffer on AgentV14 and is passed in at forward time.
    """

    DEFAULT_LAYER_CHARS = frozenset("asq")  # qa Q5: a + s,q by default

    def __init__(
        self,
        D1: int,
        D2: int,
        cond_d1_eff: int,
        cond_d2_eff: int,
        L_text: int,
        layer_type_string: str,
        extra_text_layers: str = "",
    ):
        """Initialize TextCodebookHead.

        Args:
            D1: Primary (a-layer) hidden dimension.
            D2: Secondary (s/q-layer) hidden dimension.
            cond_d1_eff: Effective conditioning dimension for d1-scale layers.
            cond_d2_eff: Effective conditioning dimension for d2-scale layers.
            L_text: Fixed text sequence length.
            layer_type_string: String of layer type chars (e.g. "aaaaasqasq").
            extra_text_layers: Additional layer chars to treat as text-active
                beyond DEFAULT_LAYER_CHARS.
        """
        super().__init__()
        self.D1 = D1
        self.D2 = D2
        self.cond_d1_eff = cond_d1_eff
        self.cond_d2_eff = cond_d2_eff
        self.L_text = L_text
        self.layer_type_string = layer_type_string
        self.extra_text_layers = extra_text_layers

        effective_chars = set(self.DEFAULT_LAYER_CHARS) | set(extra_text_layers)

        self.text_token_projs = nn.ModuleDict()
        for i, c in enumerate(layer_type_string):
            if c in effective_chars:
                out_dim = D2 if c in "sq" else D1  # mirror add_token_projs
                proj = nn.Linear(D1, out_dim)
                # Zero-init: makes the text stream's contribution to motion
                # attention exactly zero at init, so criterion 2b
                # (motion loss within 5% of v13.2) holds by construction.
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                self.text_token_projs[f"layer_{i}"] = proj

        self.text_head = nn.Linear(D1, 768)

        # Single shared 768 -> D1 projection that maps codebook lookups into
        # per-layer-projection input space. Zero-init so 's text stream
        # contributes exactly 0 to motion attention at init (criterion 2b).
        self.text_input_proj = nn.Linear(768, D1)
        nn.init.zeros_(self.text_input_proj.weight)
        nn.init.zeros_(self.text_input_proj.bias)

        self.text_pos_emb = nn.Embedding(L_text, D1)
        self.role_emb_d1 = nn.Embedding(4, cond_d1_eff)
        self.role_emb_d2 = nn.Embedding(4, cond_d2_eff)

        nn.init.zeros_(self.text_pos_emb.weight)
        nn.init.zeros_(self.role_emb_d1.weight)
        nn.init.zeros_(self.role_emb_d2.weight)

    def add_pos(self, text_stream: Tensor) -> Tensor:
        """Add learned 1D pos emb to projected text stream (B, L, D1).

        Args:
            text_stream: (B, L_text, D1) projected text token tensor.

        Returns:
            (B, L_text, D1) tensor with positional embeddings added.

        Raises:
            ValueError: If text_stream sequence length != configured L_text.
        """
        L = text_stream.shape[1]
        if L != self.L_text:
            raise ValueError(
                f"text stream L={L} != configured L_text={self.L_text}"
            )
        pos_ids = torch.arange(L, device=text_stream.device)
        return text_stream + self.text_pos_emb(pos_ids)[None]  # broadcast on B

    def project_for_layer(
        self, text_stream_d1: Tensor, layer_idx: int
    ) -> Optional[Tensor]:
        """Project (B, L, D1) text stream into per-layer width.

        Projects to D1 for 'a' layers and D2 for 's'/'q' layers.
        Returns None if this layer is not text-active.

        Args:
            text_stream_d1: (B, L_text, D1) text stream in d1 space.
            layer_idx: Index of the layer to project for.

        Returns:
            Projected tensor of shape (B, L_text, D1) or (B, L_text, D2),
            or None if layer is not text-active.
        """
        key = f"layer_{layer_idx}"
        if key not in self.text_token_projs:
            return None
        return self.text_token_projs[key](text_stream_d1)

    def forward_input(self, codebook_emb_768: Tensor) -> Tensor:
        """Project (B, L, 768) codebook lookups into D1 space and add 1D pos emb.

        Args:
            codebook_emb_768: (B, L_text, 768) normalized T5 codebook embeddings.

        Returns:
            (B, L_text, D1) tensor in D1 space with positional embeddings added.
        """
        x = self.text_input_proj(codebook_emb_768)  # (B, L_text, D1)
        return self.add_pos(x)                       # (B, L_text, D1) with text_pos_emb

    def forward_output(self, text_slice_d1: Tensor) -> Tensor:
        """Map a-layer OUTPUT text slice (B, L, D1) -> x0_pred (B, L, 768).

        Args:
            text_slice_d1: (B, L_text, D1) text slice from the last a-layer output.

        Returns:
            (B, L_text, 768) predicted codebook embedding logits.
        """
        return self.text_head(text_slice_d1)


@torch.compiler.disable
def tokenize_with_eos_fill(
    texts: List[str],
    tokenizer,  # T5TokenizerFast instance
    L_text: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize texts then EOS-fill all positions after natural EOS.

    Per prompt §4: any position beyond the natural EOS is replaced with
    eos_token_id (NOT pad_token_id). Attention mask is True everywhere
    because the filled EOS tokens are genuine content, not padding.

    Args:
        texts: List of B text strings to tokenize.
        tokenizer: T5TokenizerFast instance (or compatible HF tokenizer).
        L_text: Fixed output sequence length. Sequences are padded/truncated to this.

    Returns:
        ids: (B, L_text) torch.long token ids with post-EOS positions filled.
            Tensors are returned on CPU; the LMM wiring step must `.to(device)`.
        attention_mask: (B, L_text) torch.bool -- all True.
    """
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=L_text,
        add_special_tokens=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].long()  # (B, L_text)
    eos_id = int(tokenizer.eos_token_id)
    B, L = ids.shape
    out = ids.clone()
    for b in range(B):
        # Find first EOS in row b (T5 always emits EOS at the natural end).
        eos_pos_b = (out[b] == eos_id).nonzero(as_tuple=False)
        if eos_pos_b.numel() == 0:
            # Truncation packed L real content tokens with no EOS slot left.
            # Do NOT clobber position L-1 (that would destroy the last real
            # content token). Leave row as-is; fill loop is a no-op.
            continue
        first = int(eos_pos_b[0, 0].item())
        if first + 1 < L:
            out[b, first + 1:] = eos_id
    attn_mask = torch.ones(B, L, dtype=torch.bool)
    return out, attn_mask
