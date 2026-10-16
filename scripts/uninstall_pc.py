"""Remove startup and shortcuts; preserve private data and installation for recovery."""
import ctypes
from pathlib import Path
import sys
import winreg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if ctypes.windll.user32.MessageBoxW(None,
    "Remove KevFLIX from Windows startup and the installed-app list?\nYour settings, downloaded models and app files will be kept for recovery. Close KevFLIX first.",
    "Uninstall KevFLIX", 0x24) == 6:
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
    ctypes.windll.user32.MessageBoxW(None, "KevFLIX startup and registration removed.\nFiles were preserved at:\n" + str(ROOT), "KevFLIX", 0)
