import einops

from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


def convert_esm_weights(esm_model, cfg: HookedTransformerConfig):
    embeddings = esm_model.esm.embeddings
    state_dict = {
        "embed.embed.W_E": embeddings.word_embeddings.weight,
        "embed.pos_embed.W_pos": embeddings.position_embeddings.weight,
    }

    # Convert each encoder layer’s weights
    for l in range(cfg.n_layers):
        block = esm_model.esm.encoder.layer[l]
        state_dict[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
            block.attention.self.query.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
            block.attention.self.query.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_K"] = einops.rearrange(
            block.attention.self.key.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_K"] = einops.rearrange(
            block.attention.self.key.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_V"] = einops.rearrange(
            block.attention.self.value.weight, "(i h) m -> i m h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.b_V"] = einops.rearrange(
            block.attention.self.value.bias, "(i h) -> i h", i=cfg.n_heads
        )
        state_dict[f"blocks.{l}.attn.W_O"] = einops.rearrange(
            block.attention.output.dense.weight,
            "m (i h) -> i h m",
            i=cfg.n_heads,
        )
        state_dict[f"blocks.{l}.attn.b_O"] = block.attention.output.dense.bias
        state_dict[f"blocks.{l}.mlp.W_in"] = einops.rearrange(
            block.intermediate.dense.weight, "mlp model -> model mlp"
        )
        state_dict[f"blocks.{l}.mlp.b_in"] = block.intermediate.dense.bias
        state_dict[f"blocks.{l}.mlp.W_out"] = einops.rearrange(
            block.output.dense.weight, "model mlp -> mlp model"
        )
        state_dict[f"blocks.{l}.mlp.b_out"] = block.output.dense.bias

    # Convert the MLM head weights
    state_dict["mlm_head.W"] = esm_model.lm_head.dense.weight
    state_dict["mlm_head.b"] = esm_model.lm_head.dense.bias
    state_dict["mlm_head.ln.w"] = esm_model.lm_head.layer_norm.weight
    state_dict["mlm_head.ln.b"] = esm_model.lm_head.layer_norm.bias
    state_dict["unembed.W_U"] = embeddings.word_embeddings.weight.T
    state_dict["unembed.b_U"] = esm_model.lm_head.bias

    return state_dict
