"""Explicit source epoch correction for CHAMPS chunked evaluation.

The original cache remains unchanged. Training retains the original backward
path; evaluation accumulates metrics without attempting backward in no_grad.
"""
import hashlib


def correct_chunked_evaluation(source: bytes, expected_sha256: str) -> str:
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError("CHAMPS epoch source hash mismatch")
    text = source.decode("utf-8")
    block = '''                    if args.champs_loss:
                        raise ValueError("CHAMPS loss not supported yet with batch_chunk mode")
                    if APEX_AVAILABLE:
                        with amp.scale_loss(mb_raw_loss, opt) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        mb_raw_loss.backward()'''
    if text.count(block) != 1:
        raise ValueError("CHAMPS epoch source no longer matches reviewed correction")
    guarded = "                    if opt is not None:\n" + "\n".join("    " + line for line in block.splitlines())
    return text.replace(block, guarded)
