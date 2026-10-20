"""Install root vs persistent user profile (fix 4). Stdlib only.

The INSTALL ROOT is the versioned code/runtime folder (default %LOCALAPPDATA%\\Programs\\KevFLIX); the installer replaces
its files on update. The PROFILE ROOT holds everything personal and is never replaced by install or update:
  relay-config.json   settings: Plex account/server, media folders, UI settings, Windows-startup preference, LAN bind
  state/              job store (jobs, chunks, sessions, resume/watched, TV devices, speakers = actor identities), logs
  cache/              completed dubs and intermediate media (reused only when the fingerprints match, see config.py)
  actor-bank/         actor references owned by the app
  desktop-updates/    downloaded installers (verified before use)
  runtime/            instance registry, producer state, run specs (owned by the running instance)
  logs/
Profile location, in order: env KEVFLIX_PROFILE (absolute) > %LOCALAPPDATA%\\KevFLIX\\profile-location.json
{"profile_root": "<absolute path>"} > %LOCALAPPDATA%\\KevFLIX\\profile. Restarts reuse it. Nothing here deletes or
overwrites personal data: folders are created when missing, files are written only when absent (or atomically by
their owner), and the legacy migration copies the old config without touching the original.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

INSTALL_ROOT = Path(__file__).resolve().parents[2]
POINTER_NAME = 'profile-location.json'
LAYOUT = ('state', 'cache', 'actor-bank', 'desktop-updates', 'runtime', 'logs')
# Keys whose relative values meant "inside the install root" in the legacy layout (machine locations).
ASSET_KEYS = ('ffmpeg', 'ffprobe', 'asr_model', 'translation_model', 'translation_ct2_model', 'translation_llm_model', 'audio_event_model', 'diarization_model',
              'speaker_onnx_model', 'separation_model_dir', 'actor_runtime_manifest', 'cinematic_checkpoint',
              'cinematic_code_root', 'overlap_checkpoint', 'overlap_code_root', 'overlap_dependency_root')


def is_relative_file_path(value):
    """True only for a RELATIVE FILE PATH (has a separator or starts with '.'). A bare command name such as
    'ffmpeg' stays a PATH command and is never rewritten; absolute paths and %VARS% stay as they are."""
    if not isinstance(value, str) or not value.strip():
        return False
    expanded = os.path.expandvars(value)
    if Path(expanded).expanduser().is_absolute():
        return False
    return '/' in expanded or '\\' in expanded or expanded.startswith('.')


def publish_if_absent(tmp, target):
    """Atomically make tmp the target ONLY if target does not exist; never overwrites. Returns True if published.
    Windows: os.rename (MoveFileEx without MOVEFILE_REPLACE_EXISTING) fails when target exists. Elsewhere: hard
    link (fails if target exists) then unlink tmp. On losing the race the temp file is removed, the winner kept."""
    tmp, target = Path(tmp), Path(target)
    try:
        if os.name == 'nt':
            os.rename(tmp, target)
        else:
            os.link(tmp, target)
            tmp.unlink()
        return True
    except FileExistsError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def local_app_data():
    base = os.environ.get('LOCALAPPDATA')
    if not base:
        base = str(Path.home() / 'AppData' / 'Local')
    return Path(base)


def profile_root():
    env = os.environ.get('KEVFLIX_PROFILE', '').strip()
    if env:
        path = Path(os.path.expandvars(env)).expanduser()
        if not path.is_absolute():
            raise ValueError('KEVFLIX_PROFILE must be an absolute folder')
        return path
    pointer = local_app_data() / 'KevFLIX' / POINTER_NAME
    if pointer.is_file():
        value = json.loads(pointer.read_text(encoding='utf-8-sig')).get('profile_root', '')
        path = Path(os.path.expandvars(str(value))).expanduser()
        if not path.is_absolute():
            raise ValueError('%s must name an absolute profile_root' % pointer)
        return path
    return local_app_data() / 'KevFLIX' / 'profile'


def set_profile_root(path):
    """Choose a different profile folder (written to the per-user pointer; existing data is not moved)."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise ValueError('The profile folder must be absolute')
    pointer = local_app_data() / 'KevFLIX' / POINTER_NAME
    pointer.parent.mkdir(parents=True, exist_ok=True)
    tmp = pointer.with_name(pointer.name + '.tmp')
    tmp.write_text(json.dumps({'profile_root': str(path)}, indent=2), encoding='utf-8')
    os.replace(tmp, pointer)
    return path


class Profile:
    def __init__(self, root=None, install_root=None):
        self.root = Path(root) if root else profile_root()
        self.install_root = Path(install_root) if install_root else INSTALL_ROOT
        if self.install_root.resolve() == self.root.resolve() or self.install_root.resolve() in self.root.resolve().parents:
            raise ValueError('The profile folder must be outside the versioned install folder')

    @property
    def config(self):
        return self.root / 'relay-config.json'

    def path(self, name):
        return self.root / name

    @property
    def runtime(self):
        return self.root / 'runtime'

    @property
    def logs(self):
        return self.root / 'logs'

    def ensure(self):
        for name in LAYOUT:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        return self

    def legacy_config(self):
        return self.install_root / 'relay-config.json'

    def migrate_legacy(self, acl_from=None):
        """First start after updating from the in-install layout: COPY the old config into the profile (never
        overwrite; the original stays). Relative machine paths are made absolute against the install root; a
        relative state/cache location keeps pointing at the existing legacy data (absolute), so sessions, resume,
        watched, actors and completed dubs are reused, not moved or reset. Returns a record or None."""
        target, legacy = self.config, self.legacy_config()
        if target.exists() or not legacy.is_file() or legacy.stat().st_size == 0:
            return None
        raw = json.loads(legacy.read_text(encoding='utf-8-sig'))
        changed = {}
        for key in ASSET_KEYS:
            value = raw.get(key)
            if is_relative_file_path(value):              # 'ffmpeg' (a PATH command) is left untouched
                raw[key] = str((self.install_root / value).resolve())
                changed[key] = raw[key]
        for model in raw.get('piper_models') or []:
            if isinstance(model, dict) and is_relative_file_path(model.get('path')):
                model['path'] = str((self.install_root / model['path']).resolve())
        for key, default in (('state_dir', './state'), ('cache_dir', './state/cache')):
            value = raw.get(key, default)
            if not Path(os.path.expandvars(str(value))).expanduser().is_absolute():
                raw[key] = str((self.install_root / value).resolve())      # existing data stays where it is
                changed[key] = raw[key]
        raw.setdefault('actor_bank_dir', str(self.root / 'actor-bank'))
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name('.kevflix-migrate-%d-%d.tmp' % (os.getpid(), time.monotonic_ns()))
        try:
            with tmp.open('x', encoding='utf-8') as stream:
                if acl_from is not None:
                    acl_from(legacy, tmp)          # private ACL applied before any secret is written
                json.dump(raw, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        if not publish_if_absent(tmp, target):   # a concurrent first launch won: keep its config, never overwrite
            return None
        record = {'migrated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'from': str(legacy),
                  'original_kept': True, 'absolutized': changed}
        note = self.runtime / 'MIGRATION.json'
        note.parent.mkdir(parents=True, exist_ok=True)
        if not note.exists():
            note.write_text(json.dumps(record, indent=2), encoding='utf-8')
        return record

    def new_config(self, example, *, secrets_factory, bind, public_url, startup, media_roots):
        """Brand-new profile only (no config here and no legacy config): settings from the example with NEW local
        secrets, absolute machine paths under the install root, and profile data folders. Never called when a
        config exists, so the user's account and settings are never reset from the example."""
        if self.config.exists() or self.legacy_config().is_file():
            raise FileExistsError('A KevFLIX configuration already exists; it is kept as is')
        settings = json.loads(Path(example).read_text(encoding='utf-8-sig'))
        for name in ('api_token', 'nas_token', 'stream_secret', 'webhook_secret'):
            settings[name] = secrets_factory()
        for key in ASSET_KEYS:
            value = settings.get(key)
            if is_relative_file_path(value):              # bare command names stay PATH commands
                settings[key] = str((self.install_root / value).resolve())
        for model in settings.get('piper_models') or []:
            if isinstance(model, dict) and is_relative_file_path(model.get('path')):
                model['path'] = str((self.install_root / model['path']).resolve())
        settings.update(state_dir=str(self.root / 'state'), cache_dir=str(self.root / 'cache'),
                        actor_bank_dir=str(self.root / 'actor-bank'), bind=bind, public_url=public_url,
                        start_with_windows=bool(startup), setup_only=True)
        settings['media_roots'] = list(media_roots)
        settings.setdefault('tv', {})['roots'] = list(media_roots)
        return settings

    def write_new_config(self, settings, *, acl=None):
        """Publish a brand-new config only if none exists (atomic create-if-absent). acl(tmp) applies the private
        ACL before the secrets are written. Returns True if this call created it; False keeps the existing one."""
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.config.with_name('.kevflix-new-config-%d-%d.tmp' % (os.getpid(), time.monotonic_ns()))
        try:
            with tmp.open('x', encoding='utf-8') as stream:
                if acl is not None:
                    acl(tmp)
                json.dump(settings, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        return publish_if_absent(tmp, self.config)


def copy_if_absent(source, target):
    """Copy a file only when the target does not exist (never overwrites)."""
    source, target = Path(source), Path(target)
    if target.exists() or not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True
