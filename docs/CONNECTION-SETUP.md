# Worker connection setup

Use Start-KevFLIX-Setup.cmd when only pairing is authorized. It serves the
interface and pairing API but does not start model, media, Plex-observer or
automatic-update workers. Playback endpoints are blocked in this mode.

The TV address must identify the PC worker, including the port. The launcher
accepts a private IP with or without the HTTP prefix and supplies the default
worker port when it is omitted. It displays connection errors rather than
navigating silently to an unavailable server.

If Windows asks about network access but its approval control is unavailable,
administrator approval may be required. Allow-KevFLIX-LAN.cmd requests that
approval and runs the accompanying PowerShell helper. An administrator must
approve the operating-system prompt. No administrator credentials are included.

The helper requires an existing KevFLIX listener on a Private network. It allows
only that executable's configured TCP port, local interface and local subnet.
It does not disable the firewall or change the network category. Existing
rules with the same name are preserved.

To remove the helper's rule later, an administrator can run:

    Remove-NetFirewallRule -Name KevFLIX-Worker-Private-LAN

Connection setup is separate from movie playback, encoder validation and
model inference acceptance.
