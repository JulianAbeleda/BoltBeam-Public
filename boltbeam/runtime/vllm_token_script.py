"""Opt-in replay token forcing; uses vLLM's production request state.

All measured and reference passes must load this processor and use the same
script. This changes sampling work, so its measured cost must remain explicit.
"""
from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor


class TokenScriptProcessor(LogitsProcessor):
    """Select a scripted token without changing the model's numerical path."""

    key = "boltbeam_token_script"

    @classmethod
    def validate_params(cls, params):
        script = (params.extra_args or {}).get(cls.key)
        if script is None:
            return
        if not isinstance(script, list) or not script or any(
            type(token) is not int or token < 0 for token in script
        ):
            raise ValueError("token script must be a nonempty list of nonnegative integers")
        if params.max_tokens != len(script) or not params.ignore_eos:
            raise ValueError("token script requires ignore_eos and max_tokens equal to its length")
        if (params.structured_outputs is not None or params.allowed_token_ids is not None
                or params.bad_words or params.stop or params.stop_token_ids
                or params.thinking_token_budget is not None):
            raise ValueError("token scripts cannot override token masks, stop rules, or thinking budgets")

    def is_argmax_invariant(self):
        return False

    def __init__(self, vllm_config, req_states):
        self.speculative = vllm_config.speculative_config is not None
        self.states = req_states
        self.scripts = {}

    def add_request(self, req_idx, params):
        self.validate_params(params)
        script = (params.extra_args or {}).get(self.key)
        if script is not None and self.speculative:
            raise ValueError("token scripts require non-speculative decoding")
        self.scripts[req_idx] = tuple(script) if script is not None else None
        if script is not None and max(script) >= self.states.vocab_size:
            raise ValueError("scripted token is outside the model vocabulary")
        return script is not None

    def apply(self, logits, ctx):
        # Without speculation there is one logits row per request; host lengths
        # are exact. Read engine state, never advance an independent counter.
        for row, slot in enumerate(ctx.idx_mapping_np):
            script = self.scripts.get(int(slot))
            if script is None:
                continue
            if logits.shape[0] != len(ctx.idx_mapping_np):
                raise ValueError("token script requires one logits row per request")
            position = int(ctx.seq_lens_upper_bound_np[row] - self.states.prompt_len.np[slot])
            if position < 0:  # Intermediate prefill chunks do not emit an output token.
                continue
            if position >= len(script):
                raise ValueError("token script position outside request output")
            logits[row].fill_(float("-inf"))
            token = script[position]
            logits[row, token:token + 1].fill_(0)
        return logits
