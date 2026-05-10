"""Auto-loaded by Python startup when this file's directory is on PYTHONPATH.

Bumps the default timeout passed to torch.distributed.init_process_group so
loading 405B HF checkpoints from /lustre (slow, single-rank read) doesn't trip
the NCCL watchdog on the waiting ranks during the post-load broadcast.

Activated only when TT_DIST_TIMEOUT_MIN is set; no-op otherwise.
"""

import os

_mins = os.environ.get("TT_DIST_TIMEOUT_MIN")
if _mins:
    try:
        _mins = int(_mins)
    except ValueError:
        _mins = None

if _mins:
    try:
        from datetime import timedelta as _td
        import torch.distributed as _dist
        _orig = _dist.init_process_group

        def _patched(*args, **kwargs):
            kwargs.setdefault("timeout", _td(minutes=_mins))
            return _orig(*args, **kwargs)

        _dist.init_process_group = _patched
    except Exception:
        pass
