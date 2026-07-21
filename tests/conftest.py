"""Pytest session setup: force `transformers`/`huggingface_hub` fully offline
for the entire test session (M4 review Finding A).

Chronos-2 weights are expected already cached locally (`~/.cache/huggingface`
-- `models.py`'s module docstring, "CPU feasibility" section) from a prior
manual `chronos-forecasting` download in this dev environment. Setting these
two env vars here, at collection time, means every Chronos-2 test (real
forward passes against the cached weights, per the plan's Ground rule 7 --
they are NOT mocked) resolves purely from that local cache. A fresh clone
with an EMPTY cache gets a clear, immediate `LocalEntryNotFoundError` (from
`huggingface_hub`) the first time a Chronos-2 test tries to load the model,
instead of a silent live network call -- that error is the signal to
populate the cache once (e.g. `hf download amazon/chronos-2`, or simply
running any Chronos-2 forecast with these unset) before running the suite
offline as intended here.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
