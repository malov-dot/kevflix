"""PC-side YouTube TV host (stage 1): the real youtube.com/tv page runs in ONE background (headless) Chrome on this PC, its
screen is streamed to the KevFLIX TV app as MJPEG and the TV remote keys are sent back through the DevTools protocol.

Owner requirements, enforced here and asserted by tests/youtube/:
  * never a visible window, taskbar entry or focus steal: Chrome runs with --headless=new, CREATE_NO_WINDOW and a hidden
    startup window; keys reach the page ONLY through CDP Input.dispatchKeyEvent (no operating-system input, no global hotkeys);
  * low impact: BELOW_NORMAL priority, a Windows Job Object (dies with the server, optional memory cap), a dedicated
    --user-data-dir, one renderer process, ~20 fps screencast cap, screencast stopped while no TV client is watching,
    Chrome closed after youtube.idle_seconds without a client, ONE instance, DevTools bound to 127.0.0.1 only;
  * audio is muted and media keys / notifications / permission prompts are disabled;
  * only OUR process tree is ever killed (job object / PID), never chrome.exe by name; the module is imported lazily so the
    local/Plex startup path never loads it.
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import threading
import time
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
BOUNDARY = "kevflixframe"

# Samsung Tizen 6.5 TV browser (Chromium 108) - the first user agent tried; COBALT_UA is the console fallback.
TIZEN_UA = ("Mozilla/5.0 (SMART-TV; Linux; Tizen 6.5) AppleWebKit/537.36 (KHTML, like Gecko) "
            "108.0.5359.1/6.5 TV Safari/537.36")
COBALT_UA = ("Mozilla/5.0 (Linux armv7l) Cobalt/22.lts.4.304693-gold (unlike Gecko) v8/8.8.278.8-jit gles "
             "Starboard/13, GoogleTV_Chromecast_2020/QTS1.220805.002 (Google, Chromecast)")
USER_AGENTS = {"tizen": TIZEN_UA, "cobalt": COBALT_UA}

DEFAULT_SETTINGS = {
    "enabled": False, "profile_dir": "C:/KevFLIX-Runtime/youtube-profile", "chrome_path": "", "idle_seconds": 120,
    "grace_seconds": 5, "user_agent": "", "user_agent_profile": "tizen", "start_url": "https://www.youtube.com/tv",
    "width": 1920, "height": 1080, "jpeg_quality": 65, "every_nth_frame": 3, "max_fps": 20, "memory_limit_mb": 3072, "userscripts": [], "userscripts_when": "ready",
}
MAX_USERSCRIPT_BYTES = 4 * 1024 * 1024
CHROME_CANDIDATES = (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                     r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                     r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                     r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")

# name -> (DOM key, DOM code, keyCode, text). Samsung remote codes for the media keys: Play 415, Pause 19, PlayPause 10252,
# FF 417, RW 412; Back is 10009 (measured: with the Tizen user agent youtube.com/tv ignores Escape (27) but obeys 10009).
KEYS = {
    "up": ("ArrowUp", "ArrowUp", 38, ""), "down": ("ArrowDown", "ArrowDown", 40, ""),
    "left": ("ArrowLeft", "ArrowLeft", 37, ""), "right": ("ArrowRight", "ArrowRight", 39, ""),
    "enter": ("Enter", "Enter", 13, "\r"), "back": ("XF86Back", "XF86Back", 10009, ""),
    "backspace": ("Backspace", "Backspace", 8, ""),
    "play": ("MediaPlay", "MediaPlay", 415, ""), "pause": ("MediaPause", "MediaPause", 19, ""),
    "playpause": ("MediaPlayPause", "MediaPlayPause", 10252, ""),
    "fastforward": ("MediaFastForward", "MediaFastForward", 417, ""), "rewind": ("MediaRewind", "MediaRewind", 412, ""),
}
ALIASES = {"arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right", "select": "enter",
           "ok": "enter", "escape": "back", "mediaplaypause": "playpause", "mediaplay": "play", "mediapause": "pause",
           "mediafastforward": "fastforward", "mediarewind": "rewind", "ff": "fastforward", "rw": "rewind"}


class HostError(RuntimeError):
    pass


def key_name(name):
    value = ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if value not in KEYS:
        raise KeyError("Unknown remote key: %s" % str(name)[:20])
    return value


def key_events(name, action="press", repeat=False):
    """CDP Input.dispatchKeyEvent parameter dicts for one remote key. action: down | up | press (down then up)."""
    key, code, vk, text = KEYS[key_name(name)]
    base = {"key": key, "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk, "modifiers": 0}
    down = dict(base, type="keyDown" if text else "rawKeyDown", autoRepeat=bool(repeat))
    if text:
        down.update(text=text, unmodifiedText=text)
    up = dict(base, type="keyUp")
    if action == "down":
        return [down]
    if action == "up":
        return [up]
    if action == "press":
        return [down, up]
    raise ValueError("Key action must be down, up or press")


def mjpeg_part(jpeg):
    """One multipart/x-mixed-replace part (boundary line, headers, JPEG bytes, CRLF)."""
    return (("--%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % (BOUNDARY, len(jpeg))).encode()
            + jpeg + b"\r\n")


def mjpeg_headers():
    return {"Content-Type": "multipart/x-mixed-replace; boundary=" + BOUNDARY, "Cache-Control": "no-store, private",
            "Pragma": "no-cache", "Connection": "close", "X-Content-Type-Options": "nosniff"}


def find_chrome(configured=""):
    for path in ([configured] if configured else []) + list(CHROME_CANDIDATES):
        if path and Path(path).is_file():
            return str(path)
    raise HostError("Chrome or Edge was not found; set youtube.chrome_path")


def user_agent(settings):
    return str(settings.get("user_agent") or USER_AGENTS.get(settings.get("user_agent_profile"), TIZEN_UA))


def build_command(chrome, port, settings, profile_dir):
    """The Chrome command line. Headless (no window of any kind), muted, no media-key capture, no prompts, one renderer."""
    return [
        chrome, "--headless=new", "--remote-debugging-port=%d" % port, "--remote-debugging-address=127.0.0.1",
        "--user-data-dir=" + str(profile_dir), "--window-size=%d,%d" % (int(settings["width"]), int(settings["height"])),
        "--mute-audio", "--autoplay-policy=document-user-activation-required",
        "--disable-features=Translate,HardwareMediaKeyHandling,MediaSessionService,GlobalMediaControls",
        "--no-first-run", "--no-default-browser-check", "--disable-notifications", "--deny-permission-prompts",
        "--disable-extensions", "--disable-background-networking", "--disable-sync", "--disable-default-apps",
        "--disable-component-update", "--renderer-process-limit=1", "--noerrdialogs", "--password-store=basic",
        "--user-agent=" + user_agent(settings), "about:blank"]


def popen_kwargs():
    """Hidden, low-priority child: no console, no window (SW_HIDE), BELOW_NORMAL, its own process group."""
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt":
        info = subprocess.STARTUPINFO()
        info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = 0                                   # SW_HIDE
        kwargs.update(startupinfo=info, creationflags=CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS | CREATE_NEW_PROCESS_GROUP)
    return kwargs


class ChromeProcess:
    """Our Chrome and its children in a Windows Job Object: KILL_ON_JOB_CLOSE (dies with the server) and an optional memory cap."""

    def __init__(self, popen, job=None):
        self.popen, self.job, self.pid = popen, job, popen.pid

    def poll(self):
        return self.popen.poll()

    def kill_tree(self):
        """Terminate ONLY this process tree (job, else taskkill of our own PID)."""
        if self.job is not None and os.name == "nt":
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel.TerminateJobObject(self.job, 1)
            kernel.CloseHandle(self.job)
            self.job = None
        elif os.name == "nt" and self.popen.poll() is None:
            subprocess.run(["taskkill", "/PID", str(self.pid), "/T", "/F"], capture_output=True,
                           creationflags=CREATE_NO_WINDOW, timeout=15)
        elif self.popen.poll() is None:
            self.popen.kill()
        try:
            self.popen.wait(timeout=10)
        except Exception:
            pass

    def cpu_seconds(self):
        """Total user+kernel CPU of every process in the job (None without a job)."""
        if self.job is None or os.name != "nt":
            return None
        import ctypes
        from ctypes import wintypes

        class Accounting(ctypes.Structure):
            _fields_ = [("TotalUserTime", wintypes.LARGE_INTEGER), ("TotalKernelTime", wintypes.LARGE_INTEGER),
                        ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER), ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
                        ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                        ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        info = Accounting()
        if not kernel.QueryInformationJobObject(self.job, 1, ctypes.byref(info), ctypes.sizeof(info), None):
            return None
        return (info.TotalUserTime + info.TotalKernelTime) / 1e7


def _make_job(memory_limit_mb):
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER), ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class Counters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class Extended(ctypes.Structure):
        _fields_ = [("Basic", Basic), ("Io", Counters), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.restype = ctypes.c_void_p
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObject failed")
    info = Extended()
    info.Basic.LimitFlags = 0x2000 | 0x20                        # KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_PRIORITY_CLASS
    info.Basic.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS       # every process of the tree stays BELOW_NORMAL (Chrome cannot raise it)
    if int(memory_limit_mb) > 0:
        info.Basic.LimitFlags |= 0x200                           # JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = int(memory_limit_mb) * 1024 * 1024
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError("SetInformationJobObject failed")
    return job


def launch_chrome(command, memory_limit_mb=0):
    """Default launcher: start Chrome hidden at BELOW_NORMAL priority and put it in a kill-on-close Job Object."""
    popen = subprocess.Popen(command, **popen_kwargs())
    job = None
    if os.name == "nt":
        try:
            import ctypes
            job = _make_job(memory_limit_mb)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            if not kernel.AssignProcessToJobObject(job, int(popen._handle)):
                raise OSError("AssignProcessToJobObject failed")
        except Exception as error:                                # fall back to PID-tree kill; never fail the launch
            logging.warning("YouTube host: job object unavailable (%s)", error)
            job = None
    return ChromeProcess(popen, job)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Cdp:
    """Minimal DevTools protocol client over one aiohttp websocket."""

    def __init__(self, session, ws):
        self.session, self.ws, self.next_id, self.waiting, self.on_event = session, ws, 0, {}, None
        self.task = None

    async def send(self, method, params=None, timeout=15):
        self.next_id += 1
        number = self.next_id
        waiter = asyncio.get_running_loop().create_future()
        self.waiting[number] = waiter
        await self.ws.send_str(json.dumps({"id": number, "method": method, "params": params or {}}))
        try:
            reply = await asyncio.wait_for(waiter, timeout)
        finally:
            self.waiting.pop(number, None)
        if "error" in reply:
            raise HostError("%s: %s" % (method, reply["error"].get("message", reply["error"])))
        return reply.get("result", {})

    def post(self, method, params=None):
        """Fire and forget (screencast acks): no reply is awaited so the reader is never blocked."""
        self.next_id += 1
        asyncio.get_running_loop().create_task(
            self.ws.send_str(json.dumps({"id": self.next_id, "method": method, "params": params or {}})))

    async def read(self):
        async for message in self.ws:
            try:
                data = json.loads(message.data)
            except (TypeError, ValueError):
                continue
            if "id" in data and data["id"] in self.waiting:
                if not self.waiting[data["id"]].done():
                    self.waiting[data["id"]].set_result(data)
            elif "method" in data and self.on_event:
                self.on_event(data["method"], data.get("params", {}))
        for waiter in list(self.waiting.values()):
            if not waiter.done():
                waiter.set_exception(HostError("DevTools connection closed"))

    async def close(self):
        for part in (self.ws, self.session):
            try:
                await part.close()
            except Exception:
                pass
        if self.task:
            self.task.cancel()


class YouTubeHost:
    """One lazily started background Chrome showing youtube.com/tv. All public methods are thread-safe (the HTTP server
    is threaded); the DevTools client runs on a private asyncio loop thread that exists only while Chrome runs."""

    def __init__(self, settings=None, launcher=None, clock=time.monotonic, watch_interval=1.0):
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self._ready_scripts = []
        self._launcher, self._clock, self._watch_interval = launcher, clock, float(watch_interval)
        self._lifecycle = threading.Lock()                       # one start/stop at a time -> one instance only
        self._cond = threading.Condition()
        self._loop = self._thread = self._cdp = self._proc = None
        self._tasks = []
        self.state, self.error, self.ticket = "stopped", "", ""
        self._frame, self._counter, self._clients, self._casting = None, 0, 0, False
        self._size = (int(self.settings["width"]), int(self.settings["height"]))
        self._last_activity, self._down, self.started = clock(), set(), 0
        self.last_command, self._closing = [], False

    # ---- thread-safe API ---------------------------------------------------------------------------------------
    def status(self):
        with self._cond:
            return {"enabled": True, "state": self.state, "width": self._size[0], "height": self._size[1],
                    "frame": self._counter, "clients": self._clients, "screencast": self._casting,
                    "error": self.error, "pid": self._proc.pid if self._proc else None}

    def open(self):
        with self._lifecycle:
            if self.state == "running" and self._proc and self._proc.poll() is None:
                self.touch()
                return self._opened()
            self._shutdown_locked()
            self.state, self.error, self._closing = "starting", "", False
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, args=(self._loop,), daemon=True, name="youtube-cdp")
            self._thread.start()
            try:
                self._call(self._start(), 90)
            except BaseException as error:
                self.error = str(error) or error.__class__.__name__
                self._shutdown_locked()
                self.state = "error"
                raise HostError(self.error)
            return self._opened()

    def close(self):
        with self._lifecycle:
            self._shutdown_locked()

    def touch(self):
        self._last_activity = self._clock()

    def ticket_valid(self, ticket):
        return bool(self.ticket) and hmac.compare_digest(str(ticket or ""), self.ticket)

    def key(self, name, action="press"):
        events = key_events(name, action)
        if self.state != "running":
            raise HostError("YouTube is not open")
        self.touch()
        self._call(self._send_keys(key_name(name), events, action), 5)

    def client_enter(self):
        with self._cond:
            self._clients += 1
        self.touch()
        if self.state == "running" and self._loop:
            asyncio.run_coroutine_threadsafe(self._screencast(True), self._loop)

    def client_leave(self):
        with self._cond:
            self._clients = max(0, self._clients - 1)
        self.touch()

    def latest(self):
        with self._cond:
            return self._counter, self._frame

    def next_frame(self, after, timeout):
        """Block until a frame newer than counter `after` exists (latest-only: skipped frames are never queued).
        Returns (counter, jpeg), or None on timeout / when the host is no longer running."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._counter <= after or self._frame is None:
                if self.state != "running":
                    return None
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self._cond.wait(left)
            return self._counter, self._frame

    def cpu_seconds(self):
        return self._proc.cpu_seconds() if self._proc else None

    # ---- internals ---------------------------------------------------------------------------------------------
    def _opened(self):
        status = self.status()
        return {"state": status["state"], "width": status["width"], "height": status["height"],
                "screen_ticket": self.ticket}

    def _run_loop(self, loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
        loop.close()

    def _call(self, coroutine, timeout):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(timeout)

    def _shutdown_locked(self):
        loop, thread = self._loop, self._thread
        if loop and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._teardown(), loop).result(20)
            except Exception:
                pass
            loop.call_soon_threadsafe(loop.stop)
            if thread:
                thread.join(5)
        if self._proc:
            self._proc.kill_tree()
        self._loop = self._thread = self._cdp = self._proc = None
        self._tasks = []
        self._casting = False
        with self._cond:
            if self.state != "error":
                self.state = "stopped"
            self.ticket, self._frame = "", None
            self._cond.notify_all()

    async def _teardown(self):
        self._closing = True
        for task in self._tasks:
            task.cancel()
        if self._cdp:
            await self._cdp.close()
        if self._proc:
            self._proc.kill_tree()
        with self._cond:
            self._cond.notify_all()

    async def _start(self):
        import aiohttp
        settings = self.settings
        profile = Path(str(settings["profile_dir"]))
        profile.mkdir(parents=True, exist_ok=True)
        port = free_port()
        self.last_command = build_command(find_chrome(settings["chrome_path"]) if not self._launcher else
                                          (settings["chrome_path"] or "chrome"), port, settings, profile)
        launch = self._launcher or (lambda command, port: launch_chrome(command, settings["memory_limit_mb"]))
        self._proc = launch(self.last_command, port)
        port = getattr(self._proc, "port", port)
        base = "http://127.0.0.1:%d" % port
        session = aiohttp.ClientSession()
        target = None
        for _ in range(150):                                     # up to ~15 s for Chrome to open its DevTools port
            if self._proc.poll() is not None:
                break
            try:
                async with session.get(base + "/json/list", timeout=aiohttp.ClientTimeout(total=2)) as reply:
                    pages = [t for t in await reply.json(content_type=None) if t.get("type") == "page"]
                    if pages and pages[0].get("webSocketDebuggerUrl"):
                        target = pages[0]
                        break
            except Exception:
                pass
            await asyncio.sleep(0.1)
        if not target:
            await session.close()
            raise HostError("Chrome did not start (no DevTools page)")
        ws = await session.ws_connect(target["webSocketDebuggerUrl"], max_msg_size=0, heartbeat=20)
        self._cdp = cdp = Cdp(session, ws)
        cdp.on_event = self._on_event
        cdp.task = asyncio.ensure_future(cdp.read())
        width, height = int(settings["width"]), int(settings["height"])
        await cdp.send("Page.enable")
        try:
            await cdp.send("Page.bringToFront")                  # headless: no window exists to raise; makes the page "active"
        except HostError:
            pass
        await self._permissions(session, base)
        await self._identity(cdp)
        await self._inject(cdp, settings)
        await cdp.send("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
        self.ticket, self._size = secrets.token_urlsafe(24), (width, height)
        await cdp.send("Page.navigate", {"url": settings["start_url"]})
        self.started += 1
        self.touch()
        with self._cond:
            self.state = "running"
        await self._screencast(True)
        self._tasks = [asyncio.ensure_future(self._watch()), asyncio.ensure_future(self._watch_connection(cdp))]

    async def _inject(self, cdp, settings):
        """Run each configured userscript (youtube.userscripts: list of file paths) in the page BEFORE its own scripts, on every document.
        Third-party scripts (for example an ad blocker) live OUTSIDE this repository; a missing, unreadable or oversized file is skipped
        with a log line and never stops the page from opening."""
        self._ready_scripts = []
        for entry in settings.get("userscripts") or []:
            path = Path(str(entry))
            try:
                if not path.is_file() or path.stat().st_size > MAX_USERSCRIPT_BYTES:
                    raise OSError("missing or larger than %d bytes" % MAX_USERSCRIPT_BYTES)
                source = path.read_text(encoding="utf-8")
                if str(settings.get("userscripts_when", "ready")) == "ready":
                    # Scripts written for TizenBrew are injected after the page has loaded; some (for example TizenTube's) do nothing when run at
                    # document start. They are evaluated through DevTools when the page's DOMContentLoaded fires (page security policy cannot
                    # block that). "start" instead injects the file as is, before the page's own scripts.
                    self._ready_scripts.append(source)
                else:
                    await cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": source})
                logging.info("YouTube host: userscript injected: %s (%d bytes)", path.name, len(source))
            except (OSError, UnicodeDecodeError, HostError) as error:
                logging.warning("YouTube host: userscript skipped: %s (%s)", path.name, error)

    async def _identity(self, cdp):
        agent = user_agent(self.settings)
        major = (re.search(r"(\d+)\.\d+\.\d+\.\d+", agent) or re.search(r"Chrome/(\d+)", agent))
        major = major.group(1) if major else "108"
        full = (re.search(r"(\d+\.\d+\.\d+\.\d+)", agent) or [None, major + ".0.0.0"])[1]
        brands = [{"brand": "Chromium", "version": major}, {"brand": "Not;A=Brand", "version": "99"}]
        await cdp.send("Emulation.setUserAgentOverride", {
            "userAgent": agent, "acceptLanguage": "en-US,en;q=0.9", "platform": "Linux",
            "userAgentMetadata": {"brands": brands, "fullVersionList": [dict(b, version=full if b["brand"] == "Chromium" else "99.0.0.0") for b in brands],
                                  "fullVersion": full, "platform": "Linux", "platformVersion": "5.4.0",
                                  "architecture": "x86", "model": "", "mobile": False}})

    async def _permissions(self, session, base):
        """Best effort: deny every permission through the browser-level DevTools endpoint (the flags already deny prompts)."""
        try:
            async with session.get(base + "/json/version") as reply:
                url = (await reply.json(content_type=None)).get("webSocketDebuggerUrl")
            if not url:
                return
            async with session.ws_connect(url) as browser:
                for number, name in enumerate(("notifications", "geolocation", "midi", "audioCapture", "videoCapture"), 1):
                    await browser.send_str(json.dumps({"id": number, "method": "Browser.setPermission",
                                                       "params": {"permission": {"name": name}, "setting": "denied"}}))
                await asyncio.sleep(0.2)
        except Exception as error:
            logging.info("YouTube host: permission lockdown skipped (%s)", error)

    def _on_event(self, method, params):
        if method == "Page.domContentEventFired":
            for source in list(self._ready_scripts):
                if self._cdp:
                    self._cdp.post("Runtime.evaluate", {"expression": source, "silent": True})
            return
        if method != "Page.screencastFrame":
            return
        try:
            data = base64.b64decode(params["data"])
        except (KeyError, ValueError):
            return
        meta = params.get("metadata") or {}
        with self._cond:
            self._counter += 1
            self._frame = data
            if meta.get("deviceWidth") and meta.get("deviceHeight"):
                self._size = (int(meta["deviceWidth"]), int(meta["deviceHeight"]))
            self._cond.notify_all()
        if self._cdp and "sessionId" in params:
            self._cdp.post("Page.screencastFrameAck", {"sessionId": params["sessionId"]})   # ack EVERY frame

    async def _screencast(self, on):
        if not self._cdp or on == self._casting or self.state != "running" and on:
            return
        for attempt in range(8 if on else 1):
            try:
                if on:
                    s = self.settings
                    await self._cdp.send("Page.startScreencast", {
                        "format": "jpeg", "quality": int(s["jpeg_quality"]), "maxWidth": int(s["width"]),
                        "maxHeight": int(s["height"]), "everyNthFrame": max(1, int(s["every_nth_frame"]))})
                else:
                    await self._cdp.send("Page.stopScreencast")
                self._casting = on
                return
            except Exception as error:     # right after a navigation Chrome answers "Not attached to an active page": retry
                logging.info("YouTube host: screencast %s failed (%s)", "start" if on else "stop", error)
                await asyncio.sleep(0.5)

    async def _send_keys(self, name, events, action):
        for event in events:
            event = dict(event)
            if event["type"] != "keyUp":
                if name in self._down:
                    event["autoRepeat"] = True
                self._down.add(name)
            else:
                self._down.discard(name)
            await self._cdp.send("Input.dispatchKeyEvent", event, timeout=5)

    async def _watch(self):
        """Idle watchdog: no client for grace_seconds -> stop the screencast; no activity for idle_seconds -> close Chrome."""
        while True:
            await asyncio.sleep(self._watch_interval)
            if self._proc is None or self._proc.poll() is not None:
                break
            with self._cond:
                clients = self._clients
            idle = self._clock() - self._last_activity
            if clients == 0 and self._casting and idle >= float(self.settings["grace_seconds"]):
                await self._screencast(False)
            if clients == 0 and idle >= float(self.settings["idle_seconds"]):
                break
        threading.Thread(target=self.close, daemon=True, name="youtube-close").start()

    async def _watch_connection(self, cdp):
        try:
            await cdp.task
        except BaseException:
            pass
        if self.state == "running" and not self._closing:
            self.error = "Chrome closed"
            threading.Thread(target=self.close, daemon=True, name="youtube-close").start()


def stream_screen(host, write, max_fps=20, keepalive=2.0, should_stop=lambda: False, clock=time.monotonic):
    """Serve the latest frames as multipart parts through write(bytes) until write raises (client gone), the host stops or
    should_stop() is true. Latest-only: a slow client skips frames; the last frame is repeated every `keepalive` seconds
    on a still page so the <img> stays alive and a vanished client is noticed. Returns the number of parts written."""
    host.client_enter()
    sent, last_counter, last_sent = 0, 0, 0.0
    interval = 1.0 / max(1, int(max_fps))
    try:
        while not should_stop() and host.state == "running":
            frame = host.next_frame(last_counter, keepalive)
            if frame is None:
                frame = host.latest()
                if frame[1] is None:
                    continue
            wait = interval - (clock() - last_sent)
            if wait > 0:
                time.sleep(wait)
            newest = host.latest()
            counter, jpeg = newest if newest[1] else frame
            write(mjpeg_part(jpeg))
            last_counter, last_sent, sent = max(counter, last_counter), clock(), sent + 1
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
        pass
    finally:
        host.client_leave()
    return sent
