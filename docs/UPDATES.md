# GitHub updates

The PC can check the configured repository branch once per hour. Update requests go only to api.github.com and carry no media, network inventory, movie names or viewing context.

The updater resolves an immutable commit, downloads the three allowlisted TV-interface assets, compares their Git blob identities, and saves SHA-256 receipts. It rejects an incompatible player API version. New files are stored in a separate revision directory.

An update becomes active only when there are no TV playback sessions. Each player page uses a version-specific asset URL, so a viewing session keeps one set of files. An offline or failed check retains the current interface. A rollback endpoint on the authenticated PC API restores the previous interface and blocks that rejected revision from immediate reinstallation.

The existing updater covers **the TV interface**. Worker engine/dependency changes and the standalone launcher require a separate update/install step. This avoids silently replacing running inference code or device signatures. No automatic updates are currently activated on a TV by source publication alone.

For a public repository, no GitHub token is needed. Private repositories require a read-only token in the configured PC environment variable; it is never sent to the TV. The normal TizenBrew GitHub module loader cannot fetch a private repository.

Push UI changes to the tracked branch and update the visible version. Compatible player UI changes are picked up at a later idle check/launch. A broken UI update can still break KevFLIX itself; checksum verification proves file identity, not correctness.
