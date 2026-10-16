# GitHub updates

The PC can check the configured repository branch once per hour. Update requests go only to api.github.com and carry no media, network inventory, movie names or viewing context.

The updater resolves an immutable commit, downloads the three allowlisted TV-interface assets, compares their Git blob identities, and saves SHA-256 receipts. It rejects an incompatible player API version. New files are stored in a separate revision directory.

An update becomes active only when there are no TV playback sessions. Each player page uses a version-specific asset URL, so a viewing session keeps one set of files. An offline or failed check retains the current interface. A rollback endpoint on the authenticated PC API restores the previous interface and blocks that rejected revision from immediate reinstallation.

The separate desktop updater checks GitHub releases hourly for an installer described by desktop-release.json. It verifies size and SHA-256, then installs on the next KevFLIX start. It never interrupts active playback or stops Sem/Plex. Installation requires the product ownership marker and an idle worker lock. Private configuration/state/models are retained, replaced source is backed up, and failed installation restores replaced files. Changed Python dependencies are installed into a new virtual environment while retaining the old one for rollback. Models and their terms remain an explicit setup step. Backups use disk space until removed by the owner.

Source checkouts are not automatically replaced. Setup-only mode disables updater threads and pending desktop installation. Disable automatic updates in PC settings to stop both downloads and pending installation. The unsigned installer has not been exercised; a successful build or checksum is not runtime acceptance.

The included GitHub Actions workflow packages Windows and TV downloads on main-branch source pushes without executing the installer, application or tests. It publishes a development release tied to the commit. Increase the visible version for user-visible changes. Release assets contain generic source/runtime only. No private account, network, media, config or model files belong in a release.

TizenBrew's unversioned GitHub module URL resolves through jsDelivr and can retain an older release even after main changes. Release publication now invalidates the module metadata and legacy launcher cache paths. The module points to a versioned launcher document containing its own script and style, so old cached JavaScript cannot be paired with newer HTML. Generate that document with scripts/build_launcher.py after increasing tv/package.json and the visible product version. Previously published launcher paths are immutable.

The launcher refreshes TizenBrew's in-memory module list through its local GetModules request when opened from the original malov-dot/kevflix entry. This updates the menu metadata without adding, removing, renaming or pinning a module. If the response points to a newer KevFLIX launcher, the same entry follows its updated path while keeping the saved PC address. The request is limited to TizenBrew's localhost module context and has a bounded timeout, so offline startup can continue with the installed launcher. Standalone WGT launches skip this step. Returning to TizenBrew then reads the refreshed service cache. A page already open in memory still needs to be reopened to execute a newly published launcher.

For a public repository, no GitHub token is needed. Private repositories require a read-only token in the configured PC environment variable; it is never sent to the TV. The normal TizenBrew GitHub module loader cannot fetch a private repository.

Push UI changes to the tracked branch and update the visible version. Compatible player UI changes are picked up at a later idle check/launch. A broken UI update can still break KevFLIX itself; checksum verification proves file identity, not correctness.
