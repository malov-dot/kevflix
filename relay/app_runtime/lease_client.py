"""`relay.app_runtime.lease_client` = the pinned lease v2 client (movie-gpu-lease-v2 lease_client.py, 5cbac8a4...).

The byte-identical upstream file is relay/app_runtime/lease/lease_client.py: it resolves its bundled tolerant reader
relative to its own folder (relay/app_runtime/guard-status-read-v1/), so it must stay there. Importing this module
execs exactly those verified bytes once (leasehold.load_client: sha256 checked, no __pycache__) and makes that
module object THE `relay.app_runtime.lease_client` module, so every importer (the holder, the resident Demucs
worker, the Vox hook) shares one client module and one SOURCE_SHA256.
"""
import sys

from .leasehold import load_client as _load_client

sys.modules[__name__] = _load_client()
