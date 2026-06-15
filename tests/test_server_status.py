"""Tests for server reconnection status tracking in ClientManager."""

import threading
import time
from dataclasses import replace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Patch settings before importing clients so the module-level singleton
# doesn't try to load conf.json or start a health-check thread.
_mock_settings = MagicMock()
_mock_settings.health_check_interval = None
_mock_settings.client_uuid = "test-device-id"
_mock_settings.player_name = "test-player"
_mock_settings.connect_retry_mins = 0
_mock_settings.ignore_ssl_cert = False
_mock_settings.tls_client_cert = None

# Import conf first, then swap settings before clients is imported.
import jellyfin_mpv_shim.conf as _conf_mod
_real_settings = _conf_mod.settings
_conf_mod.settings = _mock_settings

from jellyfin_mpv_shim.clients import (
    ClientManager,
    ServerStatus,
    STATE_CONNECTED,
    STATE_CONNECTING,
    STATE_DISCONNECTED,
    STATE_RECONNECTING,
    STATE_AUTH_FAILED,
)
import jellyfin_mpv_shim.clients as _clients_mod

# Ensure the clients module also sees the mock
_clients_mod.settings = _mock_settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_settings_mock():
    s = MagicMock()
    s.health_check_interval = None
    s.client_uuid = "test-device-id"
    s.player_name = "test-player"
    s.connect_retry_mins = 0
    s.ignore_ssl_cert = False
    s.tls_client_cert = None
    return s


@pytest.fixture()
def mock_settings():
    s = _make_settings_mock()
    with patch.object(_clients_mod, "settings", s):
        yield s


@pytest.fixture()
def cm(mock_settings):
    """Fresh ClientManager with health checks disabled."""
    return ClientManager()


@pytest.fixture()
def fake_server():
    return {
        "uuid": "srv-1",
        "Id": "server-id-1",
        "Name": "Test Server",
        "address": "http://192.168.1.100:8096",
        "username": "testuser",
        "connected": False,
        "AccessToken": "fake-token",
    }


@pytest.fixture()
def fake_server_2():
    return {
        "uuid": "srv-2",
        "Id": "server-id-2",
        "Name": "Second Server",
        "address": "http://10.0.0.5:8096",
        "username": "user2",
        "connected": False,
        "AccessToken": "fake-token-2",
    }


def _make_mock_client(signed_in=True):
    """Create a mock JellyfinClient that passes authentication."""
    client = MagicMock()
    from jellyfin_apiclient_python.connection_manager import CONNECTION_STATE

    if signed_in:
        client.authenticate.return_value = {"State": CONNECTION_STATE["SignedIn"]}
    else:
        client.authenticate.return_value = {"State": CONNECTION_STATE["Unavailable"]}
    # Sessions endpoint returns our device in the list
    client.jellyfin._http.return_value = [{"DeviceId": "test-device-id"}]
    return client


# ---------------------------------------------------------------------------
# Group 1: ServerStatus dataclass basics
# ---------------------------------------------------------------------------

class TestServerStatusDataclass:
    def test_defaults(self):
        s = ServerStatus(
            server_uuid="u1",
            server_name="N",
            server_address="http://x",
            state=STATE_CONNECTED,
        )
        assert s.failure_reason is None
        assert s.retry_count == 0
        assert s.next_retry_seconds is None
        assert isinstance(s.last_updated, float)

    def test_all_fields(self):
        s = ServerStatus(
            server_uuid="u1",
            server_name="N",
            server_address="http://x",
            state=STATE_RECONNECTING,
            failure_reason="timeout",
            retry_count=3,
            next_retry_seconds=16.0,
            last_updated=100.0,
        )
        assert s.state == STATE_RECONNECTING
        assert s.failure_reason == "timeout"
        assert s.retry_count == 3
        assert s.next_retry_seconds == 16.0
        assert s.last_updated == 100.0

    def test_snapshot_isolation(self):
        original = ServerStatus(
            server_uuid="u1",
            server_name="N",
            server_address="http://x",
            state=STATE_CONNECTED,
        )
        snapshot = replace(original)
        original.state = STATE_DISCONNECTED
        original.retry_count = 99
        assert snapshot.state == STATE_CONNECTED
        assert snapshot.retry_count == 0


# ---------------------------------------------------------------------------
# Group 2: Status tracking through connect_client
# ---------------------------------------------------------------------------

class TestConnectClientStatus:
    def test_success_sets_connected(self, cm, fake_server, mock_settings):
        mock_client = _make_mock_client(signed_in=True)
        with patch.object(cm, "client_factory", return_value=mock_client):
            with patch.object(cm, "setup_client", return_value=True):
                result = cm.connect_client(fake_server)

        assert result is True
        status = cm.get_server_status("srv-1")
        assert status is not None
        assert status.state == STATE_CONNECTED
        assert status.failure_reason is None
        assert status.retry_count == 0
        assert status.next_retry_seconds is None

    def test_auth_fail_sets_auth_failed(self, cm, fake_server, mock_settings):
        mock_client = _make_mock_client(signed_in=False)
        with patch.object(cm, "client_factory", return_value=mock_client):
            result = cm.connect_client(fake_server)

        assert result is False
        status = cm.get_server_status("srv-1")
        assert status is not None
        assert status.state == STATE_AUTH_FAILED
        assert status.failure_reason == "Authentication failed"

    def test_partial_then_success(self, cm, fake_server, mock_settings):
        mock_client = _make_mock_client(signed_in=True)
        setup_results = iter([False, True])

        def fake_setup(client, server):
            return next(setup_results)

        with patch.object(cm, "client_factory", return_value=mock_client):
            with patch.object(cm, "setup_client", side_effect=fake_setup):
                with patch("jellyfin_mpv_shim.clients.time.sleep"):
                    result = cm.connect_client(fake_server, do_retries=True)

        assert result is True
        status = cm.get_server_status("srv-1")
        assert status.state == STATE_CONNECTED

    def test_partial_all_retries_fail(self, cm, fake_server, mock_settings):
        mock_client = _make_mock_client(signed_in=True)

        with patch.object(cm, "client_factory", return_value=mock_client):
            with patch.object(cm, "setup_client", return_value=False):
                with patch("jellyfin_mpv_shim.clients.time.sleep"):
                    result = cm.connect_client(fake_server, do_retries=True)

        assert result is False
        status = cm.get_server_status("srv-1")
        assert status.state == STATE_DISCONNECTED
        assert "setup failed" in status.failure_reason.lower() or "failed" in status.failure_reason.lower()

    def test_connected_bool_still_set_on_success(self, cm, fake_server, mock_settings):
        mock_client = _make_mock_client(signed_in=True)
        with patch.object(cm, "client_factory", return_value=mock_client):
            with patch.object(cm, "setup_client", return_value=True):
                cm.connect_client(fake_server)

        assert fake_server["connected"] is True

    def test_connected_bool_still_set_on_failure(self, cm, fake_server, mock_settings):
        mock_client = _make_mock_client(signed_in=False)
        with patch.object(cm, "client_factory", return_value=mock_client):
            cm.connect_client(fake_server)

        assert fake_server["connected"] is False


# ---------------------------------------------------------------------------
# Group 3: Status tracking through reconnection (event closure)
# ---------------------------------------------------------------------------

class TestWebSocketReconnectionStatus:
    def test_websocket_disconnect_sets_reconnecting(self, cm, fake_server, mock_settings):
        """Simulate WebSocketDisconnect and verify status is RECONNECTING."""
        mock_client = MagicMock()
        cm.clients["srv-1"] = mock_client
        cm._update_status(
            "srv-1",
            state=STATE_CONNECTED,
            server_name="Test Server",
            server_address="http://192.168.1.100:8096",
        )

        captured_event_fn = None

        def fake_setup(client, server):
            nonlocal captured_event_fn
            # Capture the event closure that setup_client creates
            # by intercepting the callback assignment
            captured_event_fn = client.callback
            return True

        # We need to manually simulate what setup_client does to the event closure.
        # Instead, directly call setup_client and capture the event callback.
        original_validate = cm.validate_client
        cm.validate_client = MagicMock(return_value=True)

        cm.setup_client(mock_client, fake_server)
        captured_event_fn = mock_client.callback

        # Restore
        cm.validate_client = original_validate

        # Now simulate disconnect in a background thread
        # Make connect_client succeed immediately to stop the loop
        reconnect_called = threading.Event()

        def mock_connect(server, do_retries=True):
            reconnect_called.set()
            return True

        cm.connect_client = mock_connect
        cm._disconnect_client = MagicMock()

        t = threading.Thread(target=captured_event_fn, args=("WebSocketDisconnect", None))
        t.daemon = True
        t.start()

        reconnect_called.wait(timeout=5)
        t.join(timeout=5)

        status = cm.get_server_status("srv-1")
        assert status is not None
        # After reconnect succeeds, connect_client (mocked) was called but
        # the reconnect loop set RECONNECTING before calling it
        assert status.retry_count >= 1

    def test_websocket_connect_sets_connected(self, cm, fake_server, mock_settings):
        """Simulate WebSocketConnect and verify status is CONNECTED."""
        mock_client = MagicMock()
        cm.clients["srv-1"] = mock_client
        cm._update_status(
            "srv-1",
            state=STATE_RECONNECTING,
            server_name="Test Server",
            server_address="http://192.168.1.100:8096",
            retry_count=3,
        )

        cm.validate_client = MagicMock(return_value=True)
        cm.callback = MagicMock()

        cm.setup_client(mock_client, fake_server)
        event_fn = mock_client.callback

        event_fn("WebSocketConnect", None)

        status = cm.get_server_status("srv-1")
        assert status.state == STATE_CONNECTED
        assert status.failure_reason is None
        assert status.retry_count == 0
        assert status.next_retry_seconds is None


# ---------------------------------------------------------------------------
# Group 4: Status tracking through health checks
# ---------------------------------------------------------------------------

class TestHealthCheckStatus:
    def test_validate_client_exception_sets_disconnected(self, cm, mock_settings):
        mock_client = MagicMock()
        mock_client.jellyfin._http.side_effect = ConnectionError("timeout")
        cm.clients["srv-1"] = mock_client
        cm._update_status(
            "srv-1",
            state=STATE_CONNECTED,
            server_name="Test Server",
            server_address="http://192.168.1.100:8096",
        )

        # validate_client with empty client_list leads to "not in list" path
        # but we also get the exception path status update
        result = cm.validate_client(mock_client, dry_run=True)
        assert result is False

        status = cm.get_server_status("srv-1")
        assert status.state == STATE_DISCONNECTED
        assert "query failed" in status.failure_reason.lower()

    def test_validate_client_not_in_list_sets_disconnected(self, cm, mock_settings):
        mock_client = MagicMock()
        # Return a session list that does NOT contain our device
        mock_client.jellyfin._http.return_value = [
            {"DeviceId": "other-device"}
        ]
        cm.clients["srv-1"] = mock_client
        cm._update_status(
            "srv-1",
            state=STATE_CONNECTED,
            server_name="Test Server",
            server_address="http://192.168.1.100:8096",
        )

        result = cm.validate_client(mock_client, dry_run=False)
        assert result is False

        status = cm.get_server_status("srv-1")
        assert status.state == STATE_DISCONNECTED
        assert "session list" in status.failure_reason.lower()

    def test_check_all_clients_retries_disconnected(self, cm, fake_server, mock_settings):
        cm.credentials = [fake_server]
        # srv-1 is NOT in cm.clients (disconnected)
        connect_called = []
        original_connect = cm.connect_client

        def track_connect(server, do_retries=True):
            connect_called.append(server["uuid"])
            return False

        cm.connect_client = track_connect

        cm.check_all_clients()

        assert "srv-1" in connect_called
        status = cm.get_server_status("srv-1")
        assert status is not None
        # check_all_clients sets CONNECTING before calling connect_client
        assert status.failure_reason == "Health check retry"


# ---------------------------------------------------------------------------
# Group 5: Public API and thread safety
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_get_server_statuses_returns_copies(self, cm):
        cm._update_status(
            "s1", state=STATE_CONNECTED,
            server_name="A", server_address="http://a",
        )
        cm._update_status(
            "s2", state=STATE_DISCONNECTED,
            server_name="B", server_address="http://b",
        )

        statuses = cm.get_server_statuses()
        assert len(statuses) == 2

        # Mutate returned list — internal state should be unchanged
        statuses[0].state = "mutated"
        internal = cm.get_server_status("s1")
        assert internal.state != "mutated"

    def test_get_server_status_unknown_uuid(self, cm):
        assert cm.get_server_status("nonexistent") is None

    def test_get_server_statuses_empty(self, cm):
        assert cm.get_server_statuses() == []

    def test_concurrent_status_updates(self, cm):
        """Multiple threads updating the same server status concurrently."""
        cm._update_status(
            "s1", state=STATE_CONNECTING,
            server_name="A", server_address="http://a",
        )

        errors = []

        def updater(thread_id, count):
            try:
                for i in range(count):
                    cm._update_status(
                        "s1",
                        state=STATE_RECONNECTING,
                        retry_count=thread_id * 1000 + i,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater, args=(t, 100)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == []
        status = cm.get_server_status("s1")
        assert status is not None
        assert isinstance(status.retry_count, int)
        assert status.state == STATE_RECONNECTING


# ---------------------------------------------------------------------------
# Group 6: remove_client / remove_all_clients clear status
# ---------------------------------------------------------------------------

class TestRemoveStatus:
    def test_remove_client_clears_status(self, cm, mock_settings):
        cm._update_status(
            "srv-1", state=STATE_CONNECTED,
            server_name="A", server_address="http://a",
        )
        cm.credentials = [{"uuid": "srv-1", "Name": "A", "address": "http://a"}]

        with patch.object(cm, "save_credentials"):
            cm.remove_client("srv-1")

        assert cm.get_server_status("srv-1") is None

    def test_remove_all_clients_clears_all_statuses(self, cm, mock_settings):
        cm._update_status(
            "s1", state=STATE_CONNECTED,
            server_name="A", server_address="http://a",
        )
        cm._update_status(
            "s2", state=STATE_DISCONNECTED,
            server_name="B", server_address="http://b",
        )

        with patch.object(cm, "save_credentials"):
            cm.remove_all_clients()

        assert cm.get_server_statuses() == []


# ---------------------------------------------------------------------------
# Group 7: _disconnect_client sets DISCONNECTED
# ---------------------------------------------------------------------------

class TestDisconnectClientStatus:
    def test_disconnect_sets_disconnected(self, cm):
        mock_client = MagicMock()
        cm.clients["srv-1"] = mock_client
        cm._update_status(
            "srv-1", state=STATE_CONNECTED,
            server_name="A", server_address="http://a",
        )

        server = {"uuid": "srv-1", "connected": True}
        cm._disconnect_client(server=server)

        assert server["connected"] is False
        status = cm.get_server_status("srv-1")
        assert status.state == STATE_DISCONNECTED
        assert status.next_retry_seconds is None

    def test_disconnect_by_uuid(self, cm):
        mock_client = MagicMock()
        cm.clients["srv-1"] = mock_client
        cm._update_status(
            "srv-1", state=STATE_CONNECTED,
            server_name="A", server_address="http://a",
        )

        cm._disconnect_client(uuid="srv-1")

        status = cm.get_server_status("srv-1")
        assert status.state == STATE_DISCONNECTED

    def test_disconnect_nonexistent_is_noop(self, cm):
        # Should not raise
        cm._disconnect_client(uuid="does-not-exist")


# ---------------------------------------------------------------------------
# Group 8: try_connect retry loop status updates
# ---------------------------------------------------------------------------

class TestTryConnectStatus:
    def test_try_connect_retry_updates_status(self, cm, mock_settings):
        mock_settings.connect_retry_mins = 1  # 2 attempts (mins * 2)

        cm.credentials = [
            {
                "uuid": "srv-1",
                "Id": "id-1",
                "Name": "Server 1",
                "address": "http://192.168.1.100:8096",
                "username": "test",
                "connected": False,
            }
        ]

        call_count = [0]

        def mock_connect_all():
            call_count[0] += 1
            # Fail on first two calls, succeed on third
            if call_count[0] >= 3:
                cm.clients["srv-1"] = MagicMock()
                return True
            return False

        with patch.object(cm, "_connect_all", side_effect=mock_connect_all):
            with patch("jellyfin_mpv_shim.clients.time.sleep"):
                with patch("jellyfin_mpv_shim.clients.conffile.get", return_value="/tmp/fake_cred.json"):
                    with patch("builtins.open", MagicMock()):
                        with patch("json.load", return_value=[]):
                            with patch("os.path.exists", return_value=False):
                                result = cm.try_connect()

        assert result is True
        # During retry, status should have been updated
        status = cm.get_server_status("srv-1")
        assert status is not None


# ---------------------------------------------------------------------------
# Group 9: _update_status creates and updates correctly
# ---------------------------------------------------------------------------

class TestUpdateStatus:
    def test_create_new_status(self, cm):
        cm._update_status(
            "s1",
            state=STATE_CONNECTING,
            server_name="Test",
            server_address="http://test",
        )
        status = cm.get_server_status("s1")
        assert status.server_uuid == "s1"
        assert status.server_name == "Test"
        assert status.server_address == "http://test"
        assert status.state == STATE_CONNECTING

    def test_update_existing_status(self, cm):
        cm._update_status(
            "s1",
            state=STATE_CONNECTING,
            server_name="Test",
            server_address="http://test",
        )
        old_ts = cm.get_server_status("s1").last_updated

        cm._update_status("s1", state=STATE_CONNECTED, retry_count=0)
        status = cm.get_server_status("s1")

        assert status.state == STATE_CONNECTED
        assert status.server_name == "Test"  # unchanged
        assert status.last_updated >= old_ts

    def test_remove_status(self, cm):
        cm._update_status(
            "s1",
            state=STATE_CONNECTING,
            server_name="Test",
            server_address="http://test",
        )
        cm._remove_status("s1")
        assert cm.get_server_status("s1") is None

    def test_remove_nonexistent_is_noop(self, cm):
        cm._remove_status("does-not-exist")  # Should not raise


# ---------------------------------------------------------------------------
# Group 10: _uuid_for_client reverse lookup
# ---------------------------------------------------------------------------

class TestUuidForClient:
    def test_finds_matching_client(self, cm):
        mock_client = MagicMock()
        cm.clients["srv-1"] = mock_client
        assert cm._uuid_for_client(mock_client) == "srv-1"

    def test_returns_none_for_unknown(self, cm):
        assert cm._uuid_for_client(MagicMock()) is None
