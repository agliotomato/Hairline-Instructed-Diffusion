from __future__ import annotations
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F

class MaskedCrossAttnProcessor:
    def __init__(self, mask_pyramid: Dict[int, torch.Tensor], target_indices: List[int], scaling_factor: float = 1.0):
        self.mask_pyramid = mask_pyramid
        self.target_indices = target_indices
        self.scaling_factor = scaling_factor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Standard Attention Calculation
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # attention_probs shape: (batch_size * heads, query_len, key_len)

        # --- MASK INJECTION logic ---
        if self.target_indices and encoder_hidden_states.shape[1] > max(self.target_indices):
            # Check resolution to pick correct mask
            dim = int(sequence_length ** 0.5)
            
            if dim in self.mask_pyramid:
                 # mask: (1, 1, dim, dim) -> (1, 1, dim*dim) -> tranpose -> (1, dim*dim, 1)
                 mask = self.mask_pyramid[dim].view(1, -1, 1).to(attention_probs.device)
                 
                 for idx in self.target_indices:
                     # Suppress pixels outside mask (where mask is 0)
                     attention_probs[:, :, idx] *= mask.squeeze(-1)

        # ----------------------------

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def get_token_indices(tokenizer, prompt: str, trigger_word: str):
    if not trigger_word:
        return []
    
    input_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True).input_ids
    trigger_ids = tokenizer(trigger_word, add_special_tokens=False).input_ids
    
    indices = []
    len_trigger = len(trigger_ids)
    if len_trigger == 0:
        return []
        
    for i in range(len(input_ids) - len_trigger + 1):
        if input_ids[i : i + len_trigger] == trigger_ids:
            indices.extend(list(range(i, i + len_trigger)))
            
    return list(set(indices))

def create_mask_pyramid(mask_tensor: torch.Tensor, max_res: int = 64) -> Dict[int, torch.Tensor]:
    # mask_tensor: [1, 1, H, W]
    pyramid = {}
    current_res = max_res
    while current_res >= 8:
        m = F.interpolate(mask_tensor, size=(current_res, current_res), mode='nearest')
        pyramid[current_res] = m
        current_res //= 2
    return pyramid
