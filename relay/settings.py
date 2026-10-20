"""Private PC settings and Plex account linking; no credentials are returned to the TV."""
from __future__ import annotations
import ctypes
import ipaddress
import json
import os
import secrets
import sys
import subprocess
import threading
import time
from pathlib import Path
from urllib import request, parse, error
from . import __version__
from .config import Config, gpu_first_migration
from .plex import NoCrossOriginRedirect, Plex
from .plex_browse import PlexBrowse

ROOT = Path(__file__).resolve().parent.parent


def preserve_private_acl(source, target):
    if os.name != "nt":
        os.chmod(target, 0o600)
        return
    from ctypes import wintypes
    api = ctypes.WinDLL("advapi32", use_last_error=True)
    api.GetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID,
                                    wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    api.SetFileSecurityW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID]
    needed = wintypes.DWORD()
    api.GetFileSecurityW(str(source), 4, None, 0, ctypes.byref(needed))
    if not needed.value:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_string_buffer(needed.value)
    if not api.GetFileSecurityW(str(source), 4, buffer, needed.value, ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not api.SetFileSecurityW(str(target), 0x80000004, buffer):
        raise ctypes.WinError(ctypes.get_last_error())


def _legacy_run_value():
    """The old HKCU Run entry (earlier builds): a command, or None. It was enumerated at login on one PC but never launched."""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as key:
            command, _ = winreg.QueryValueEx(key, "KevFLIX")
            return command if str(ROOT / "scripts" / "run_worker.py").lower() in command.lower() else None
    except OSError:
        return None


def _remove_legacy_run_value():
    import winreg
    if _legacy_run_value() is None:
        return
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_SET_VALUE) as key:
        try:
            winreg.DeleteValue(key, "KevFLIX")
        except FileNotFoundError:
            pass


def startup_enabled():
    """True when the Windows logon tasks (relay/startup_tasks.py) exist."""
    if os.name != "nt":
        return False
    from . import startup_tasks
    try:
        return startup_tasks.enabled()
    except (OSError, subprocess.SubprocessError):
        return False


def set_startup(enabled):
    """Start the worker (and the tray icon) at Windows login through scheduled tasks; the old Run entry is always removed."""
    if os.name != "nt":
        if enabled:
            raise ValueError("Windows startup registration is available only on Windows")
        return
    from . import startup_tasks
    if enabled:
        startup_tasks.install(ROOT)
        _remove_legacy_run_value()
    else:
        startup_tasks.remove()
        _remove_legacy_run_value()


class PCSettings:
    def __init__(self, app):
        self.app, self.config = app, app.config
        self.lock = threading.RLock()
        self.pending = {}
        self.servers = {}
        self.setup_process = None
        self.opener = request.build_opener(NoCrossOriginRedirect)

    def raw(self):
        return json.loads(self.config.file.read_text(encoding="utf-8-sig"))

    def save_raw(self, value):
        gpu_first_migration(value)       # app-owned CPU-only movie keys only (relay/config.py); nothing else changes
        temporary = self.config.file.with_name(".kevflix-config-" + secrets.token_hex(8) + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                preserve_private_acl(self.config.file, temporary)
                json.dump(value, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            validated = Config(temporary)
            # All relative paths are resolved from the same directory as the live config.
            os.replace(temporary, self.config.file)
            locked = self.config.data.get("setup_locked", False)
            self.config.data = validated.data
            self.config.data["setup_locked"] = locked
            if locked:
                self.config.data["setup_only"] = True
            self.app.plex.plex = Plex(self.config)
            self.app.tv.plex = self.app.plex.plex
            self.app.tv.catalog = PlexBrowse(self.app.plex.plex)
        finally:
            if temporary.exists():
                temporary.unlink()

    def idle_required(self):
        with self.app.tv.lock:
            if self.app.tv.sessions:
                raise ValueError("Stop the KevFLIX playback session before changing PC settings")

    def engine_idle_required(self):
        self.idle_required()
        if self.setup_process and self.setup_process.poll() is None:
            raise ValueError("Wait for the component installation to finish before changing engine settings")
        if (self.app.jobs.active_job is not None or self.app.jobs.active_owner is not None
                or self.app.publications or self.app.jobs.publication_owners):
            raise ValueError("Finish or cancel active movie work before changing the dubbing engine or source language")
        if any(job["state"] not in ("complete", "error", "cancelled", "needs_review")
               for job in self.app.store.jobs()):
            raise ValueError("Finish or cancel queued movie jobs before changing the dubbing engine or source language")

    def engine_status(self):
        """Read existing coordinator observations only; never ensure/start/load a service from settings."""
        status = getattr(self.app.jobs, "engine_status", None)
        if not callable(status):
            return {"engine": self.config["speech_engine"], "state": "unobserved"}
        try:
            return status()
        except Exception as error:
            return {"engine": self.config["speech_engine"], "state": "status_error", "error": str(error)}

    def view(self):
        data = self.config.data
        return {"bind": data["bind"], "port": data["port"], "public_url": data["public_url"],
                "media_roots": data["media_roots"], "path_mappings": data["path_mappings"],
                "device": data["device"], "cpu_threads": data["cpu_threads"],
                "speech_engine": data["speech_engine"], "xt_source_language": data.get("xt_source_language", "auto"),
                "engine_status": self.engine_status(),
                "tv": {k: data["tv"][k] for k in ("max_height","video_encoder","upscale","hdr_mode","buffer_segments")},
                "updates_enabled": data["updates"]["enabled"], "start_with_windows": startup_enabled(),
                "setup_only": data["setup_only"], "setup_locked": data.get("setup_locked", False),
                "setup_running": bool(self.setup_process and self.setup_process.poll() is None),
                "plex": {"linked": bool(data.get("plex_account_token")),
                         "configured": bool(data["plex"]["token"]), "url": data["plex"]["url"],
                         "server_name": data.get("plex_server_name", "")},
                # Read-only: translation_engine / translation_llm_model are edited in relay-config.json (a Qwen3-8B
                # folder is a 16 GB user-supplied path, so it is deliberately not a remotely editable setting).
                "translation_engine": data.get("translation_engine", "m2m100"),
                "assets": {key: Path(data[key]).is_dir() for key in
                           (("asr_model","diarization_model","separation_model_dir","translation_llm_model")
                            if data.get("translation_engine") == "qwen3" else
                            ("asr_model","translation_model","diarization_model","separation_model_dir"))}}

    def save(self, supplied):
        self.idle_required()
        with self.lock:
            raw = self.raw()
            if "speech_engine" in supplied:
                if supplied["speech_engine"] not in ("generic", "xtranslator", "actor", "sapi", "piper"):
                    raise ValueError("Choose generic, xtranslator, actor, sapi or piper")
                raw["speech_engine"] = supplied["speech_engine"]
            if "xt_source_language" in supplied:
                if supplied["xt_source_language"] not in ("auto", "ru", "es", "en"):
                    raise ValueError("Choose automatic detection, Russian, Spanish or English")
                raw["xt_source_language"] = supplied["xt_source_language"]
            engine_change = (raw.get("speech_engine", self.config["speech_engine"]) != self.config["speech_engine"]
                             or raw.get("xt_source_language", "auto") != self.config.data.get("xt_source_language", "auto"))
            if engine_change:
                self.engine_idle_required()
            for name in ("bind", "public_url"):
                if name in supplied:
                    raw[name] = str(supplied[name]).strip()
            if "port" in supplied:
                raw["port"] = int(supplied["port"])
                if not 1024 <= raw["port"] <= 65535:
                    raise ValueError("Choose a worker port between 1024 and 65535")
            if "media_roots" in supplied:
                roots = supplied["media_roots"]
                if not isinstance(roots, list) or not roots or len(roots) > 20 or not all(isinstance(x, str) and x.strip() for x in roots):
                    raise ValueError("Provide at least one media folder")
                raw["media_roots"] = roots
                raw["tv"] = {**raw.get("tv", {}), "roots": roots}
            if "path_mappings" in supplied:
                mappings = supplied["path_mappings"]
                if not isinstance(mappings, list) or len(mappings) > 40:
                    raise ValueError("Invalid path mappings")
                for entry in mappings:
                    if not isinstance(entry, dict) or set(entry) != {"nas", "pc"} or not all(isinstance(x,str) and x.strip() for x in entry.values()):
                        raise ValueError("Each path mapping needs nas and pc paths")
                raw["path_mappings"] = mappings
            if "device" in supplied:
                # GPU-first: the movie path always uses 'cuda'. The obsolete processor selector values ('cpu',
                # 'cooperative-cuda', empty) no longer select anything and are not stored; any other value is refused.
                if supplied["device"] == "cuda":
                    raw["device"] = "cuda"
                elif supplied["device"] not in ("cpu", "cooperative-cuda", "", None):
                    raise ValueError("Movie dubbing runs on the GPU; the processor setting accepts only cuda")
            if "cpu_threads" in supplied:
                raw["cpu_threads"] = max(1, min(16, int(supplied["cpu_threads"])))
            if "tv" in supplied:
                allowed = ("max_height","video_encoder","upscale","hdr_mode","buffer_segments")
                values = {k: supplied["tv"][k] for k in allowed if k in supplied["tv"]}
                if "hdr_mode" in values and values["hdr_mode"] not in ("auto","sdr"):
                    raise ValueError("Choose auto or sdr HDR mode")
                if "buffer_segments" in values:
                    values["buffer_segments"] = max(1, min(6, int(values["buffer_segments"])))
                raw["tv"] = {**raw.get("tv", {}), **values}
            if "updates_enabled" in supplied:
                raw["updates"] = {**raw.get("updates", {}), "enabled": bool(supplied["updates_enabled"])}
                raw["desktop_updates"] = {**raw.get("desktop_updates", {}), "enabled": bool(supplied["updates_enabled"])}
            if "start_with_windows" in supplied:
                raw["start_with_windows"] = bool(supplied["start_with_windows"])
            if "setup_only" in supplied:
                if self.config.data.get("setup_locked") and not supplied["setup_only"]:
                    raise ValueError("This worker was launched with the setup-only safety lock")
                raw["setup_only"] = bool(supplied["setup_only"])
            old_address = (self.config["bind"], self.config["port"], self.config["setup_only"])
            if engine_change:
                # Close the old producer before this Config-validated atomic save, then rebuild lazily.
                # Failures propagate rather than returning a false "applied" response.
                self.app.jobs.reconfigure_engine(lambda: self.save_raw(raw))
            else:
                self.save_raw(raw)
            if "start_with_windows" in supplied:
                set_startup(bool(supplied["start_with_windows"]))
            result = self.view()
            result["restart_required"] = old_address != (self.config["bind"], self.config["port"], self.config["setup_only"])
            return result

    def install_tools(self, kind, accepted=False):
        if os.name != "nt" or kind not in ("ffmpeg", "dependencies", "models", "xtranslator"):
            raise ValueError("Choose a Windows processing component")
        if kind == "xtranslator":
            self.engine_idle_required()
        else:
            self.idle_required()
        with self.lock:
            if self.setup_process and self.setup_process.poll() is None:
                raise ValueError("A component installation is already running")
            if kind in ("models", "xtranslator") and not accepted:
                raise ValueError("Read THIRD-PARTY.md and accept the upstream model terms first")
            if kind == "xtranslator":
                args = [sys.executable, "-m", "relay.app_runtime.xt_provision", "--config", str(self.config.file),
                        "--distro", self.config["xt_wsl_distro"]]
            elif kind == "ffmpeg":
                args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "Setup-FFmpeg.ps1")]
            else:
                args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "Setup-Worker.ps1")]
                if kind == "models":
                    args += ["-DownloadModels", "-AcceptModelTerms"]
            log_path = self.config.state / "component-install.log"
            env = dict(os.environ)
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PIP_REQUIRE_VIRTUALENV"):
                env.pop(key, None)
            with log_path.open("ab") as log:
                self.setup_process = subprocess.Popen(args, cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
            return {"started": True, "component": kind,
                    "message": "Downloading installation files. No playback or model inference is run. See state/component-install.log for progress."}

    def plex_request(self, path, method="GET", token=None):
        raw = self.raw()
        client = raw.get("plex_client_identifier")
        if not client:
            client = secrets.token_hex(16)
            raw["plex_client_identifier"] = client
            self.save_raw(raw)
        headers = {"Accept": "application/json", "X-Plex-Product": "KevFLIX",
                   "X-Plex-Client-Identifier": client, "X-Plex-Version": __version__}
        if token:
            headers["X-Plex-Token"] = token
        req = request.Request("https://plex.tv" + path, method=method, headers=headers,
                              data=b"" if method == "POST" else None)
        try:
            with self.opener.open(req, timeout=15) as response:
                return json.load(response)
        except error.HTTPError as exc:
            raise ValueError("Plex account service returned HTTP " + str(exc.code)) from None
        except error.URLError:
            raise ValueError("Plex account sign-in needs an internet connection. Local playback can remain offline.") from None

    def link_start(self):
        with self.lock:
            pin = self.plex_request("/api/v2/pins?strong=true", "POST")
            ident = str(pin["id"])
            self.pending = {k: v for k,v in self.pending.items() if v["expires"] > time.time()}
            self.pending[ident] = {"code": pin["code"], "expires": time.time() + int(pin.get("expiresIn",1800))}
            client = self.raw()["plex_client_identifier"]
            query = parse.urlencode({"clientID": client, "code": pin["code"],
                                      "context[device][product]": "KevFLIX"})
            return {"id": ident, "url": "https://app.plex.tv/auth#?" + query}

    def link_status(self, ident):
        with self.lock:
            pin = self.pending.get(str(ident))
            if not pin or pin["expires"] < time.time():
                raise ValueError("Plex sign-in expired; start again")
            value = self.plex_request("/api/v2/pins/" + str(ident) + "?" + parse.urlencode({"code": pin["code"]}))
            token = value.get("authToken")
            if not token:
                return {"linked": False}
            raw = self.raw()
            raw["plex_account_token"] = token
            self.save_raw(raw)
            del self.pending[str(ident)]
            return {"linked": True}

    def resources(self):
        token = self.raw().get("plex_account_token")
        if not token:
            raise ValueError("Connect your Plex account first")
        values = self.plex_request("/api/v2/resources?includeHttps=1&includeRelay=0", token=token)
        result, saved = [], {}
        for item in values:
            if "server" not in str(item.get("provides", "")).split(","):
                continue
            connections = []
            for connection in item.get("connections", []):
                try:
                    local = ipaddress.ip_address(connection["address"]).is_private
                except (ValueError, KeyError):
                    local = False
                if local and not connection.get("relay"):
                    connections.append(connection)
            if not connections:
                continue
            connections.sort(key=lambda c: (c.get("protocol") != "http", not c.get("local", False)))
            connection = connections[0]
            ident = str(item["clientIdentifier"])
            url = str(connection["uri"])
            saved[ident] = {"token": item.get("accessToken") or token, "url": url,
                            "name": item.get("name", "Plex server")}
            result.append({"id": ident, "name": saved[ident]["name"], "url": url})
        self.servers = saved
        return result

    def select_server(self, ident):
        self.idle_required()
        with self.lock:
            if ident not in self.servers:
                self.resources()
            value = self.servers.get(ident)
            if not value:
                raise ValueError("Choose one of the locally reachable Plex servers")
            raw = self.raw()
            raw["plex"] = {**raw.get("plex", {}), "url": value["url"], "token": value["token"]}
            raw["plex_server_name"] = value["name"]
            self.save_raw(raw)
            return self.view()

    def disconnect(self):
        self.idle_required()
        with self.lock:
            raw = self.raw()
            raw.pop("plex_account_token", None)
            raw.pop("plex_server_name", None)
            raw["plex"] = {**raw.get("plex", {}), "token": ""}
            self.save_raw(raw)
            self.servers = {}
            return self.view()

    def plex_folders(self):
        return [{"id":str(item["key"]), "title":item["title"],
                 "paths":[x["path"] for x in item.get("Location",[]) if x.get("path")]}
                for item in self.app.plex.plex.sections() if item.get("type") == "movie"]
