"""Remove startup and shortcuts; preserve the user profile (settings, library state, dubs) and the installation."""
import ctypes
from pathlib import Path
import sys
import winreg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from relay.app_runtime.paths import Profile  # noqa: E402

profile = Profile(install_root=ROOT)
if ctypes.windll.user32.MessageBoxW(None,
    "Remove KevFLIX from Windows startup and the installed-app list?\nYour settings, library state, completed dubs and "
    "downloaded models are kept in:\n" + str(profile.root) + "\nThe app files are kept for recovery.",
    "Uninstall KevFLIX", 0x24) == 6:
    from relay.app_runtime import instances
    stopped = instances.stop(profile, timeout_s=60)          # only a validated live KevFLIX instance is signalled
    from relay.settings import set_startup
    set_startup(False)
    # Resolve the actual Windows Desktop, including redirected Desktop folders.
    buffer = ctypes.create_unicode_buffer(32768)
    if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buffer) == 0:
        shortcut = Path(buffer.value) / "KevFLIX PC.lnk"
        if shortcut.is_file():
            shortcut.unlink()
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Uninstall\KevFLIX")
    except FileNotFoundError:
        pass
    note = {"stopped": "KevFLIX was stopped.", "no active instance": "KevFLIX was not running."}.get(
        stopped.get("result"), "KevFLIX could not be stopped (" + str(stopped.get("reason")) + "); close it manually.")
    ctypes.windll.user32.MessageBoxW(None, "KevFLIX startup and registration removed. " + note +
        "\nApp files were preserved at:\n" + str(ROOT) + "\nYour profile was preserved at:\n" + str(profile.root),
        "KevFLIX", 0)
