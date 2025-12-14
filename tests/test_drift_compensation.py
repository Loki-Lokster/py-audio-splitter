import configparser

import numpy as np

from tests.fakes import FakeDeviceInfo


def _make_manager(tmp_path, audio_manager_module, *, latency1=0.07, latency2=0.0):
    audio_manager = audio_manager_module(
        devices=[
            FakeDeviceInfo(name="CABLE Output (VB-Audio Virtual Cable)", maxInputChannels=2, defaultSampleRate=48000.0),
            FakeDeviceInfo(name="Speakers (Realtek(R) Audio)", maxOutputChannels=2, defaultSampleRate=48000.0),
            FakeDeviceInfo(name="Headphones (GSA 70 Main Audio)", maxOutputChannels=2, defaultSampleRate=48000.0),
        ]
    )

    cfg_path = tmp_path / "settings.cfg"
    cfg = configparser.ConfigParser()
    cfg["Devices"] = {
        "device_1": "Speakers (Realtek(R) Audio)",
        "device_2": "Headphones (GSA 70 Main Audio)",
    }
    cfg["Settings"] = {
        "device_count": "2",
        "device_1_volume": "1.0",
        "device_1_latency": str(latency1),
        "device_2_volume": "1.0",
        "device_2_latency": str(latency2),
        "channels": "2",
        "frames_per_buffer": "64",
        "sample_rate": "1000",
        "base_buffer": "0.25",
        "drift_kp": "0.02",
        "drift_max_rate": "0.001",
        "drift_log_interval": "1",
        "drift_persist": "2",
    }
    with cfg_path.open("w", encoding="utf-8", newline="\n") as f:
        cfg.write(f)

    manager = audio_manager.AudioManager(config_path=str(cfg_path))
    manager._channels = 2
    manager._sample_rate = 1000
    manager._frames_per_buffer = 64
    manager._load_drift_compensation_settings()
    manager._setup_ring_buffer()
    manager._setup_playout_states()
    return manager


def test_initial_read_pos_uses_silent_preroll(tmp_path, audio_manager_module):
    manager = _make_manager(tmp_path, audio_manager_module, latency1=0.07, latency2=0.0)

    # write_pos starts at 0, so read_pos should be negative by target delay frames (base_buffer + latency).
    base = int(manager._base_buffer_seconds * manager._sample_rate)
    d1_target = base + int(manager.devices["device_1"].latency * manager._sample_rate)
    d2_target = base + int(manager.devices["device_2"].latency * manager._sample_rate)

    assert manager._device_state["device_1"].read_pos == -float(d1_target)
    assert manager._device_state["device_2"].read_pos == -float(d2_target)


def test_rate_adjust_sign_and_clamp(tmp_path, audio_manager_module):
    manager = _make_manager(tmp_path, audio_manager_module, latency1=0.07, latency2=0.0)
    cb = manager._make_output_callback("device_1")

    # Fill ring buffer with some samples and advance write_pos.
    manager._ring_write(np.zeros((512, manager._channels), dtype=np.float32))

    state = manager._device_state["device_1"]
    device = manager.devices["device_1"]

    base = int(manager._base_buffer_seconds * manager._sample_rate)
    target = base + int(device.latency * manager._sample_rate)

    # Make lag too high by +100 frames => +0.1s error => kp=0.02 => 0.002, clamped to 0.001
    manager._write_pos = 10_000
    state.read_pos = float(manager._write_pos - target - 100)
    out_bytes, _ = cb(None, 64, None, None)
    assert len(out_bytes) > 0
    assert state.rate_adjust == manager._max_rate_adjust
    assert device.clock_drift_seconds is not None

    # Make lag too low by -100 frames => -0.1s error => -0.002, clamped to -0.001
    state.read_pos = float(manager._write_pos - target + 100)
    out_bytes, _ = cb(None, 64, None, None)
    assert len(out_bytes) > 0
    assert state.rate_adjust == -manager._max_rate_adjust


def test_drift_logging_is_throttled_and_persistent(tmp_path, audio_manager_module, monkeypatch):
    manager = _make_manager(tmp_path, audio_manager_module, latency1=0.07, latency2=0.0)
    cb = manager._make_output_callback("device_1")

    # Provide enough audio to avoid underflow resync.
    manager._ring_write(np.zeros((4096, manager._channels), dtype=np.float32))

    base = int(manager._base_buffer_seconds * manager._sample_rate)
    device = manager.devices["device_1"]
    target = base + int(device.latency * manager._sample_rate)
    state = manager._device_state["device_1"]

    # Force a large drift error (~300ms).
    manager._write_pos = 20_000
    state.read_pos = float(manager._write_pos - target - 300)

    t = {"now": 0.0}

    def fake_time():
        return t["now"]

    import scripts.audio_manager as am

    monkeypatch.setattr(am.time, "time", fake_time)

    # 0s: should not log yet (needs persist=2s)
    cb(None, 64, None, None)
    assert manager.current_log == ""

    # 1s: still no log
    t["now"] = 1.0
    cb(None, 64, None, None)
    assert manager.current_log == ""

    # 2.1s: should log once
    t["now"] = 2.1
    cb(None, 64, None, None)
    assert "Drift:" in manager.current_log

