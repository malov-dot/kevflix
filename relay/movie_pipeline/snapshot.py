"""Private, consistent snapshots of the job Store for speculative window preparation (relay.movie_pipeline, W10; moved from movie-trial-v1/pipeline).

The analysis worker prepares a window (ASR, rescue, speakers, translation, plan) while the coordinator renders another
window. Preparation reads and WRITES the store: SpeakerBank.bind/attach_reference, the reference_vectors cache and the
v4 window intent. Those writes must not reach the shared job store: actor ids are committed only by the coordinator
(actors.commit_actors). So preparation runs on a private copy:

  snapshot_store(real_store, dest_dir, Store, fence=, check_fence=) -> a relay.store.Store subclass over
  <dest_dir>/relay.sqlite3. The copy is made with the sqlite3 online backup API from the real file, giving a consistent
  point-in-time copy with WAL included.

The copy contains the job row and the claimed chunk row ('working', same claim token), so the v4 chunk fence holds in
the snapshot, and ActorPipeline.process accepts coordinator._ClaimedStore(snapshot, job, idx, token).

Stale writers. Every transaction the snapshot opens, including every v4 fenced_write of the plan and receipt files,
first re-checks the SAME fence against the REAL store (beta_line_ledger.check_fence, read-only). So a preparation whose
claim was cancelled, revised or superseded raises Superseded at its next store access or file commit. The coordinator
also never admits a new claim of a chunk while any worker of an older claim of that chunk is still running
(TrialCoordinator busy set), so no stale worker shares a chunk folder with a new claim.
"""
from __future__ import annotations

import contextlib
import shutil
import sqlite3
from pathlib import Path


def snapshot_store(real_store, dest_dir, store_class, *, fence=None, check_fence=None):
    dest_dir = Path(dest_dir)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    target = dest_dir / 'relay.sqlite3'
    source = sqlite3.connect(str(real_store.path), timeout=30)
    try:
        source.execute('PRAGMA busy_timeout=30000')
        copy = sqlite3.connect(str(target))
        try:
            source.backup(copy)
        finally:
            copy.close()
    finally:
        source.close()

    class SnapshotStore(store_class):
        def __init__(self, directory):
            self._trial_real, self._trial_fence, self._trial_check = real_store, fence, check_fence
            super().__init__(directory)

        @contextlib.contextmanager
        def connect(self):
            if self._trial_fence is not None and self._trial_check is not None:
                self._trial_check(self._trial_real, self._trial_fence)     # raises Superseded when the claim ended
            with super().connect() as db:
                yield db

    return SnapshotStore(dest_dir)


def discard(dest_dir):
    shutil.rmtree(Path(dest_dir), ignore_errors=True)
