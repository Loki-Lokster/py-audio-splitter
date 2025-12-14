from scripts.cli_repl import CliRepl


class DummyManager:
    def __init__(self):
        self.calls = []
        self.devices = {}
        self._configured = ["Speakers"]

    def update_device_settings(self, device_name, volume=None, latency=None, persist=True):
        self.calls.append(("update_device_settings", device_name, volume, latency, persist))

    def set_stream_setting(self, key, value, persist=True):
        self.calls.append(("set_stream_setting", key, value, persist))

    def restart_audio(self):
        self.calls.append(("restart_audio",))

    @staticmethod
    def list_available_devices():
        return ["Speakers", "Headphones"]

    @staticmethod
    def list_available_input_devices():
        return ["Mic"]

    def get_configured_output_devices(self):
        return list(self._configured)

    def set_output_devices(self, device_names, persist=True):
        self.calls.append(("set_output_devices", list(device_names), persist))
        self._configured = list(device_names)

    @staticmethod
    def get_runtime_info():
        return {
            "version": "0.0.0",
            "config_path": "settings.cfg",
            "input_device_name": "Mic",
            "sample_rate": "48000",
            "channels": "2",
            "frames_per_buffer": "2048",
            "format": "float32",
            "log": "",
        }

    @staticmethod
    def get_output_device_rows():
        return [{"key": "device_1", "name": "Speakers", "volume": "1.00", "latency_ms": "0", "drift_ms": "+0"}]


def test_repl_set_volume_calls_manager():
    mgr = DummyManager()
    repl = CliRepl(mgr)
    res = repl._dispatch("set volume 1 0.5")
    assert res.message.startswith("Updated device_1 volume")
    assert mgr.calls[0][:3] == ("update_device_settings", "device_1", 0.5)


def test_repl_set_input_restarts_streams():
    mgr = DummyManager()
    repl = CliRepl(mgr)
    res = repl._dispatch('set input "CABLE Output"')
    assert "restarted streams" in res.message
    assert mgr.calls[0] == ("set_stream_setting", "input_device", "CABLE Output", True)
    assert mgr.calls[1] == ("restart_audio",)


def test_repl_add_output_appends_and_restarts():
    mgr = DummyManager()
    repl = CliRepl(mgr)
    res = repl._dispatch("add output 2")
    assert "Added output" in res.message
    assert mgr.calls[0][0] == "set_output_devices"
    assert mgr.calls[1] == ("restart_audio",)

def test_repl_add_output_by_name_match():
    mgr = DummyManager()
    repl = CliRepl(mgr)
    res = repl._dispatch('add output "Head"')
    assert "Added output" in res.message
    assert any("Headphones" in x for x in mgr.calls[0][1])


def test_repl_remove_output_removes_and_restarts():
    mgr = DummyManager()
    mgr._configured = ["Speakers", "Headphones"]
    repl = CliRepl(mgr)
    res = repl._dispatch("remove output 2")
    assert "Removed output" in res.message
    assert mgr.calls[0][0] == "set_output_devices"
    assert mgr.calls[1] == ("restart_audio",)


def test_repl_remove_output_by_name_match():
    mgr = DummyManager()
    mgr._configured = ["Speakers", "Headphones"]
    repl = CliRepl(mgr)
    res = repl._dispatch('remove output "Head"')
    assert "Removed output" in res.message
    assert mgr._configured == ["Speakers"]
