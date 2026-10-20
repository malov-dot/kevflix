"""Loaded only through the XT supervisor's private PYTHONPATH. Runtime is offline."""
import ipaddress
import os
import socket
import sys


_original_bind = socket.socket.bind


def _loopback_port_probe_bind(sock, address):
    # vLLM 0.14 probes TCP ports with bind(('', 0)) or ('', VLLM_PORT).
    # Narrow wildcard binding before the original bind emits its audit event;
    # the original bind still validates the supplied port without coercion.
    if sock.type == socket.SOCK_STREAM and isinstance(address, tuple) and len(address) >= 2:
        if sock.family == socket.AF_INET and len(address) == 2 and address[0] in ('', '0.0.0.0'):
            address = ('127.0.0.1',) + address[1:]
        elif sock.family == socket.AF_INET6 and len(address) in (2, 3, 4) and address[0] in ('', '::'):
            address = ('::1',) + address[1:]  # retain IPv6 flowinfo/scopeid exactly
    return _original_bind(sock, address)


def _local(address):
    if isinstance(address, str):
        return address.startswith('/') or address.startswith('\0')  # Unix-domain sockets
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _audit(event, args):
    if event in {'socket.connect', 'socket.bind'} and not _local(args[1]):
        raise RuntimeError('XT offline policy refuses non-loopback socket: %r' % (args[1],))
    if event == 'socket.getaddrinfo' and args[0] not in {None, 'localhost', '127.0.0.1', '::1'}:
        raise RuntimeError('XT offline policy refuses external DNS')
    if event == 'subprocess.Popen':
        executable = os.path.basename(str(args[0])).lower()
        if executable in {'curl', 'wget', 'pip', 'pip3', 'huggingface-cli', 'hf'}:
            raise RuntimeError('XT runtime cannot download or install assets')


socket.socket.bind = _loopback_port_probe_bind
sys.addaudithook(_audit)
os.environ['KEVFLIX_XT_OFFLINE_GUARD'] = '1'
