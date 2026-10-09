"""Pure token-based generation diagnostics for future evaluation runs.

Historical round-two ``generation_cap_hits`` only checked tokenizer-EOS
absence. They can contain false positives and cannot be exactly corrected
without the original generated token IDs. Saved-text accuracy is unaffected.
"""

from numbers import Integral
from collections.abc import Sequence


def hit_generation_cap(generated_token_ids: Sequence[int], max_new_tokens: int,
                       eos_token_id: int | Sequence[int] | None) -> bool:
    """Return whether a continuation exhausts its budget without configured EOS.

    Pass continuation IDs only, excluding the prompt. Use the actual per-row
    continuation length; remove batch padding if it is not an EOS token. For
    EOS-terminated rows, retained EOS tokens make the result false even when
    the batch tensor is padded to the length limit. ``None`` or an empty EOS
    sequence means no EOS stopping token is configured.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if eos_token_id is None:
        stopping_ids = frozenset()
    elif isinstance(eos_token_id, Integral) and not isinstance(eos_token_id, bool):
        stopping_ids = frozenset((int(eos_token_id),))
    else:
        if isinstance(eos_token_id, (str, bytes)) or not isinstance(eos_token_id, Sequence):
            raise TypeError("eos_token_id must be an integer, integer sequence, or None")
        if any(isinstance(token, bool) or not isinstance(token, Integral) for token in eos_token_id):
            raise TypeError("every EOS token ID must be an integer")
        stopping_ids = frozenset(int(token) for token in eos_token_id)
    return len(generated_token_ids) >= max_new_tokens and not any(
        int(token) in stopping_ids for token in generated_token_ids)
