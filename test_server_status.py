"""
Verify ServerStatus tracking logic in ClientManager without a live Jellyfin server.

Tests:
1. _update_server_status creates/updates ServerStatus entries.
2. connect_client records FAILED on auth exception.
3. connect_client records CONNECTED on success path.
4. get_server_statuses returns a serializable snapshot.
5. check_all_clients updates status for disconnected servers.
"""

import sys
import os
import time
import uuid

# Ensure the package is importable from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Stub out optional/heavy deps that clients.py imports at module level.
import types

# Stub jellyfin_apiclient_python
fake_jf = types.ModuleType("jellyfin_apiclient_python")


class FakeJellyfinClient:
    """Minimal stand-in for JellyfinClient used by connect_client."""

    def __init__(self, **kw):
        self.config = types.SimpleNamespace(data={}, app=lambda *a, **k: None)
        self.auth = types.SimpleNamespace(
            connect_to_address=lambda *a: None,
            login=lambda *a, **k: {},
            credentials=types.SimpleNamespace(get_credentials=lambda: {"Servers": [{}]}),
            create_session_with_client_auth=lambda: None,
        )
        self.jellyfin = types.SimpleNamespace(
            post_capabilities=lambda *a, **k: None,
            _http=lambda *a, **k: [],
        )
        self.callback = None
        self.callback_ws = None

    def authenticate(self, payload, discover=False):
        # Simulate auth failure by default; tests override via monkey-patch.
        raise ConnectionError("Simulated auth failure")

    def start(self, **kw):
        pass

    def stop(self):
        pass


fake_jf.JellyfinClient = FakeJellyfinClient

fake_cm = types.ModuleType("jellyfin_apiclient_python.connection_manager")
fake_cm.CONNECTION_STATE = {"SignedIn": "SignedIn"}
fake_jf.connection_manager = fake_cm

sys.modules["jellyfin_apiclient_python"] = fake_jf
sys.modules["jellyfin_apiclient_python.connection_manager"] = fake_cm

# Stub other optional deps that may not be installed
for mod_name in [
    "pystray",
    "PIL",
    "PIL.Image",
    "discord",
    "pywebview",
    "jinja2",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

# Now import the module under test
from jellyfin_mpv_shim.clients import (
    ClientManager,
    ServerStatus,
    SERVER_STATUS_CONNECTED,
    SERVER_STATUS_CONNECTING,
    SERVER_STATUS_RECONNECTING,
    SERVER_STATUS_FAILED,
    SERVER_STATUS_DISCONNECTED,
)


def test_update_server_status():
    cm = ClientManager.__new__(ClientManager)
    cm.server_statuses = {}
    cm.is_stopping = False

    server = {"uuid": "test-uuid-1", "address": "http://fake:8096"}

    cm._update_server_status(server, SERVER_STATUS_CONNECTING, increment_attempt=True)
    ss = cm.server_statuses["test-uuid-1"]
    assert ss.status == SERVER_STATUS_CONNECTING
    assert ss.attempt_count == 1
    assert ss.address == "http://fake:8096"

    cm._update_server_status(
        server,
        SERVER_STATUS_RECONNECTING,
        last_error="WebSocket disconnected",
        retry_wait_seconds=4,
        increment_attempt=True,
    )
    assert ss.status == SERVER_STATUS_RECONNECTING
    assert ss.last_error == "WebSocket disconnected"
    assert ss.retry_wait_seconds == 4
    assert ss.attempt_count == 2

    cm._update_server_status(server, SERVER_STATUS_CONNECTED)
    assert ss.status == SERVER_STATUS_CONNECTED
    assert ss.retry_wait_seconds == 0  # cleared on connected

    print("PASS: test_update_server_status")


def test_connect_client_auth_exception():
    cm = ClientManager.__new__(ClientManager)
    cm.server_statuses = {}
    cm.clients = {}
    cm.usernames = {}
    cm.credentials = []
    cm.is_stopping = False
    cm.callback = lambda *a: None

    server = {"uuid": "uuid-fail", "address": "http://fail:8096"}

    result = cm.connect_client(server, do_retries=False)
    assert result is False
    ss = cm.server_statuses["uuid-fail"]
    assert ss.status == SERVER_STATUS_FAILED
    assert "ConnectionError" in ss.last_error or "Simulated" in ss.last_error
    print("PASS: test_connect_client_auth_exception")


def test_connect_client_success():
    cm = ClientManager.__new__(ClientManager)
    cm.server_statuses = {}
    cm.clients = {}
    cm.usernames = {}
    cm.credentials = []
    cm.is_stopping = False
    cm.callback = lambda *a: None

    server = {"uuid": "uuid-ok", "address": "http://ok:8096", "username": "alice"}

    # Monkey-patch client_factory to return a client that succeeds.
    class SuccessClient(FakeJellyfinClient):
        def authenticate(self, payload, discover=False):
            return {"State": "SignedIn"}

    cm.client_factory = lambda: SuccessClient()

    # Patch validate_client to always say connected.
    cm.validate_client = lambda client, dry_run=False: True

    # Patch setup_client to simulate successful setup (store client, return True).
    original_setup = cm.setup_client

    def fake_setup(client, srv):
        cm.clients[srv["uuid"]] = client
        return True

    cm.setup_client = fake_setup

    result = cm.connect_client(server, do_retries=False)
    assert result is True
    ss = cm.server_statuses["uuid-ok"]
    assert ss.status == SERVER_STATUS_CONNECTED
    print("PASS: test_connect_client_success")


def test_get_server_statuses_snapshot():
    cm = ClientManager.__new__(ClientManager)
    cm.server_statuses = {}
    cm.is_stopping = False

    server = {"uuid": "snap-1", "address": "http://snap:8096"}
    cm._update_server_status(server, SERVER_STATUS_FAILED, last_error="timeout")

    snapshot = cm.get_server_statuses()
    assert isinstance(snapshot, list)
    assert len(snapshot) == 1
    entry = snapshot[0]
    assert entry["uuid"] == "snap-1"
    assert entry["status"] == SERVER_STATUS_FAILED
    assert entry["last_error"] == "timeout"
    assert isinstance(entry["last_check_time"], float)
    print("PASS: test_get_server_statuses_snapshot")


def test_check_all_clients_retries_disconnected():
    cm = ClientManager.__new__(ClientManager)
    cm.server_statuses = {}
    cm.clients = {}
    cm.usernames = {}
    cm.is_stopping = False
    cm.callback = lambda *a: None

    # A server that is in credentials but NOT in clients (disconnected).
    server = {"uuid": "retry-1", "address": "http://retry:8096", "Id": "s1"}
    cm.credentials = [server]

    # Patch connect_client to always fail so we can verify FAILED status is set.
    def fail_connect(srv, do_retries=True):
        cm._update_server_status(srv, SERVER_STATUS_FAILED, last_error="Still down")
        return False

    cm.connect_client = fail_connect

    cm.check_all_clients()

    ss = cm.server_statuses["retry-1"]
    assert ss.status == SERVER_STATUS_FAILED
    assert ss.last_error == "Still down"
    assert ss.attempt_count >= 1
    print("PASS: test_check_all_clients_retries_disconnected")


if __name__ == "__main__":
    test_update_server_status()
    test_connect_client_auth_exception()
    test_connect_client_success()
    test_get_server_statuses_snapshot()
    test_check_all_clients_retries_disconnected()
    print("\nAll tests passed.")
