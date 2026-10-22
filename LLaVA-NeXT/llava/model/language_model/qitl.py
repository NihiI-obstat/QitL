"""QitL (Question-in-the-Loop): training-free, question-aware visual-token pruning for LLaVA-Video (Qwen2 backbone).

A copy of the question is placed after every chunk of GROUP_SIZE frames:

    [prompt text] [V_1] [Q] [V_2] [Q] ... [V_C] [Q] [question text, assistant prompt]

Token types: TEXT=0, VISUAL=1, QUERY=2. Each frame contributes rows x (cols patches + 1 newline) visual
tokens (13 x 14 = 182 for LLaVA-Video). Attention rules during prefill:

- visual tokens never see query copies;
- a query copy sees the prompt text and only the visual tokens of its own chunk; before pruning,
  copies of different chunks do not see each other;
- the trailing text (the real question and the assistant prompt) sees no query copy until
  TEXT_GATE_LAYER, and all of them from that layer on.
"""

import math

import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

GROUP_SIZE = 8  
PRUNE_LAYER = 6  
TEXT_GATE_LAYER = 20  

TEXT, VISUAL, QUERY = 0, 1, 2


def interleave_query(frames, q_embeds):
    """frames: (F, T, D) per-frame visual tokens; q_embeds: (Lq, D). Returns (embeds, token_types)."""
    parts, types = [], []
    for f in range(frames.shape[0]):
        parts.append(frames[f])
        types.append(torch.full((frames.shape[1],), VISUAL, dtype=torch.long, device=q_embeds.device))
        if (f + 1) % GROUP_SIZE == 0 or f == frames.shape[0] - 1:
            parts.append(q_embeds)
            types.append(torch.full((q_embeds.shape[0],), QUERY, dtype=torch.long, device=q_embeds.device))
    return torch.cat(parts, dim=0), torch.cat(types)


def _chunks(token_types):
    """[(vis_start, vis_end, q_start, q_end)] per chunk, inclusive bounds, from a 1-D token-type list."""
    chunks, i, n = [], 0, len(token_types)
    while i < n:
        if token_types[i] != VISUAL:
            i += 1
            continue
        vs = i
        while i < n and token_types[i] == VISUAL:
            i += 1
        qs = i
        while i < n and token_types[i] == QUERY:
            i += 1
        assert i > qs, "every visual run must be followed by a question copy"
        chunks.append((vs, qs - 1, qs, i - 1))
    return chunks


def _attention_mask(token_types, chunks, before_prune, gate_open, dtype):
    """Additive (1, 1, N, N) mask implementing the attention rules in the module docstring."""
    n, device = token_types.shape[0], token_types.device
    chunk_id = torch.full((n,), -1, dtype=torch.long, device=device)
    for c, (vs, _, _, qe) in enumerate(chunks):
        chunk_id[vs : qe + 1] = c
    vis, query = token_types == VISUAL, token_types == QUERY
    other_chunk = chunk_id[:, None] != chunk_id[None, :]
    allowed = torch.ones(n, n, dtype=torch.bool, device=device).tril()
    allowed &= ~(vis[:, None] & query[None, :])
    allowed &= ~(query[:, None] & vis[None, :] & other_chunk)
    if before_prune:
        allowed &= ~(query[:, None] & query[None, :] & other_chunk)
    if not gate_open:
        trailing = (token_types == TEXT) & (torch.arange(n, device=device) > chunks[-1][3])
        allowed &= ~(trailing[:, None] & query[None, :])
    mask = torch.zeros(1, 1, n, n, dtype=dtype, device=device)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)


def _extend_rope(layers, max_pos, device, dtype):
    """After pruning, position ids are sparse and can exceed the key/value length of a layer; make every
    rotary embedding return its full cos/sin table instead of truncating it to that length."""
    for layer in layers:
        rope = layer.self_attn.rotary_emb
        if max_pos > rope.max_seq_len_cached:
            rope._set_cos_sin_cache(seq_len=max_pos, device=device, dtype=dtype)
        if not hasattr(rope, "_original_forward"):
            rope._original_forward = rope.forward

            def full_table(x, seq_len=None, rope=rope):
                if seq_len is not None and seq_len > rope.max_seq_len_cached:
                    rope._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
                return rope.cos_cached.to(dtype=x.dtype), rope.sin_cached.to(dtype=x.dtype)

            rope.forward = full_table


@torch.no_grad()
def _select(layer, pre_hidden, post_hidden, token_types, chunks, position_ids, tokens_per_frame, ratio):
    """Pick the tokens kept after PRUNE_LAYER. Returns (hidden, position_ids, token_types) of the new sequence."""
    attn = layer.self_attn
    n, head_dim = pre_hidden.shape[1], attn.head_dim
    normed = layer.input_layernorm(pre_hidden)
    q_all = attn.q_proj(normed).view(1, n, attn.num_heads, head_dim).transpose(1, 2)
    k_all = attn.k_proj(normed).view(1, n, attn.num_key_value_heads, head_dim).transpose(1, 2)
    cos, sin = attn.rotary_emb(k_all, seq_len=n)
    q_all, k_all = apply_rotary_pos_emb(q_all, k_all, cos, sin, position_ids)
    k_all = repeat_kv(k_all, attn.num_heads // attn.num_key_value_heads)

    side = int(round((math.sqrt(1 + 4 * tokens_per_frame) - 1) / 2))  # rows = cols; each row ends with a newline
    assert side * (side + 1) == tokens_per_frame
    keep = set((token_types == TEXT).nonzero(as_tuple=True)[0].tolist())
    residuals = []  # (query-token position, residual hidden state, residual position id) per chunk
    for vs, ve, _, qe in chunks:
        keep.add(qe)
        q_vec = q_all[0, :, qe, :]
        chunk_patches = []
        for fs in range(vs, ve + 1, tokens_per_frame):
            frame = range(fs, min(fs + tokens_per_frame, ve + 1))
            patches = [p for j, p in enumerate(frame) if j % (side + 1) != side]
            keep.update(p for j, p in enumerate(frame) if j % (side + 1) == side)  # newlines: always kept
            idx = torch.tensor(patches, dtype=torch.long, device=pre_hidden.device)
            scores = torch.matmul(q_vec.unsqueeze(1), k_all[0, :, idx, :].transpose(-2, -1)).squeeze(1) / math.sqrt(head_dim)
            n_keep = min(max(1, round(len(patches) * ratio)), len(patches))
            keep.update(patches[i] for i in scores.mean(dim=0).topk(n_keep)[1].tolist())
            chunk_patches += patches

        idx = torch.tensor(chunk_patches, dtype=torch.long, device=pre_hidden.device)
        logits = torch.matmul(q_vec.unsqueeze(1), k_all[0, :, idx, :].transpose(-2, -1)).squeeze(1) / math.sqrt(head_dim)
        rank, prob = logits.mean(dim=0), torch.softmax(logits.float(), dim=-1).mean(dim=0)
        kept = torch.tensor([p in keep for p in chunk_patches], dtype=torch.bool, device=pre_hidden.device)
        if kept.sum() > 1 and not kept.all():
            drop = torch.where(kept, rank, torch.full_like(rank, float("inf"))).argmin().item()
            kept[drop] = False
            keep.discard(chunk_patches[drop])
            w, dropped = prob[~kept], idx[~kept]
            res_vec = (w[:, None] * post_hidden[0, dropped].float()).sum(0) / w.sum()
            res_pos = int(torch.round((w * dropped.float()).sum() / w.sum()).item())
            residuals.append((qe, res_vec.to(post_hidden.dtype), res_pos))

    keep_idx = torch.tensor(sorted(keep), dtype=torch.long, device=pre_hidden.device)
    slot = {p: i for i, p in enumerate(keep_idx.tolist())}
    h_parts, pos_parts, tt_parts, start = [], [], [], 0
    for qe, res_vec, res_pos in sorted(residuals, key=lambda r: slot[r[0]]):  # [kept tokens; residual; query token]
        end = slot[qe]
        h_parts += [post_hidden[:, keep_idx[start:end], :], res_vec.view(1, 1, -1)]
        pos_parts += [keep_idx[start:end], torch.tensor([res_pos], dtype=torch.long, device=pre_hidden.device)]
        tt_parts += [token_types[keep_idx[start:end]], torch.tensor([VISUAL], dtype=torch.long, device=pre_hidden.device)]
        start = end
    h_parts.append(post_hidden[:, keep_idx[start:], :])
    pos_parts.append(keep_idx[start:])
    tt_parts.append(token_types[keep_idx[start:]])
    return torch.cat(h_parts, dim=1), torch.cat(pos_parts).unsqueeze(0), torch.cat(tt_parts)


@torch.no_grad()
def qitl_forward(lm, inputs_embeds, token_types, tokens_per_frame, ratio, position_ids=None):
    """Prefill of LlavaQwenForCausalLM `lm` with QitL pruning (batch size 1). Returns logits and the cache."""
    assert inputs_embeds.shape[0] == 1, "QitL evaluation runs with batch size 1"
    model = lm.get_model()
    n, device, dtype = inputs_embeds.shape[1], inputs_embeds.device, inputs_embeds.dtype
    token_types = token_types[0]
    chunks = _chunks(token_types.tolist())
    if position_ids is None:
        position_ids = torch.arange(n, dtype=torch.long, device=device).unsqueeze(0)
    _extend_rope(model.layers, n + 256, device, dtype)

    mask = _attention_mask(token_types, chunks, before_prune=True, gate_open=False, dtype=dtype)
    cache, hidden = DynamicCache(), inputs_embeds
    for idx, layer in enumerate(model.layers):
        if idx == TEXT_GATE_LAYER:
            mask = _attention_mask(token_types, chunks, before_prune=False, gate_open=True, dtype=dtype)
        pre_hidden = hidden
        hidden = layer(hidden, attention_mask=mask, position_ids=position_ids, past_key_value=cache, output_attentions=False, use_cache=True)[0]
        if idx == PRUNE_LAYER:
            hidden, position_ids, token_types = _select(layer, pre_hidden, hidden, token_types, chunks, position_ids, tokens_per_frame, ratio)
            chunks = _chunks(token_types.tolist())
            mask = _attention_mask(token_types, chunks, before_prune=False, gate_open=False, dtype=dtype)

    logits = lm.lm_head(model.norm(hidden)).float()
    return CausalLMOutputWithPast(logits=logits, past_key_values=cache)
