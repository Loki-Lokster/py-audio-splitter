import configparser

from tests.fakes import FakeDeviceInfo


def test_config_utf8_roundtrip(tmp_path, audio_manager_module):
    audio_manager = audio_manager_module(
        devices=[
            # Input: VB Cable
            FakeDeviceInfo(name="CABLE Output (VB-Audio Virtual Cable)", maxInputChannels=2, defaultSampleRate=48000.0),
            # Output: non-ASCII name
            FakeDeviceInfo(name="Наушники (USB Audio)", maxOutputChannels=2, defaultSampleRate=48000.0),
        ]
    )

    cfg_path = tmp_path / "settings.cfg"
    cfg = configparser.ConfigParser()
    cfg["Devices"] = {"device_1": "Наушники (USB Audio)"}
    cfg["Settings"] = {"device_count": "1", "device_1_volume": "1.0", "device_1_latency": "0.0"}
    with cfg_path.open("w", encoding="utf-8", newline="\n") as f:
        cfg.write(f)

    manager = audio_manager.AudioManager(config_path=str(cfg_path))
    assert "device_1" in manager.devices

    # Ensure saving preserves UTF-8 and doesn't crash.
    manager.save_config()
    text = cfg_path.read_text(encoding="utf-8")
    assert "Наушники" in text


def test_find_device_index_fuzzy_matching(tmp_path, audio_manager_module):
    audio_manager = audio_manager_module(
        devices=[
            FakeDeviceInfo(name="Headphones (GSA 70 Main Audio)", maxOutputChannels=2),
            FakeDeviceInfo(name="Speakers (Realtek(R) Audio)", maxOutputChannels=2),
        ]
    )

    cfg_path = tmp_path / "settings.cfg"
    cfg_path.write_text(
        "\n".join(
            [
                "[Devices]",
                "device_1 = Headphones (GSA 70 Main Audio)",
                "",
                "[Settings]",
                "device_count = 1",
                "device_1_volume = 1.0",
                "device_1_latency = 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manager = audio_manager.AudioManager(config_path=str(cfg_path))
    idx = manager._find_device_index("Headphones")
    assert idx == 0
    idx2 = manager._find_device_index("Speakers (Realtek(R) Audio)")
    assert idx2 == 1


def test_list_available_devices_dedupes_and_skips(audio_manager_module):
    audio_manager = audio_manager_module(
        devices=[
            FakeDeviceInfo(name="Microsoft Sound Mapper - Output", maxOutputChannels=2),
            FakeDeviceInfo(name="Primary Sound Driver", maxOutputChannels=2),
            FakeDeviceInfo(name="Speakers (Realtek(R) Audio)", maxOutputChannels=2),
            # Simulate a truncated duplicate name
            FakeDeviceInfo(name="Speakers (Realtek(R) Aud", maxOutputChannels=2),
        ]
    )

    devices = audio_manager.AudioManager.list_available_devices()
    assert "Microsoft Sound Mapper - Output" not in devices
    assert "Primary Sound Driver" not in devices
    assert "Speakers (Realtek(R) Audio)" in devices
    assert "Speakers (Realtek(R) Aud" not in devices


def test_find_input_device_prefers_vb_cable(tmp_path, audio_manager_module):
    audio_manager = audio_manager_module(
        devices=[
            FakeDeviceInfo(name="Microphone Array (Realtek(R) Audio)", maxInputChannels=2, defaultSampleRate=44100.0),
            FakeDeviceInfo(name="CABLE Output (VB-Audio Virtual Cable)", maxInputChannels=2, defaultSampleRate=48000.0),
            FakeDeviceInfo(name="Speakers (Realtek(R) Audio)", maxOutputChannels=2, defaultSampleRate=48000.0),
        ]
    )

    cfg_path = tmp_path / "settings.cfg"
    cfg_path.write_text(
        "\n".join(
            [
                "[Devices]",
                "device_1 = Speakers (Realtek(R) Audio)",
                "",
                "[Settings]",
                "device_count = 1",
                "device_1_volume = 1.0",
                "device_1_latency = 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manager = audio_manager.AudioManager(config_path=str(cfg_path))
    assert manager._find_input_device_index() == 1


def test_get_status_string_does_not_require_devices_section(tmp_path, audio_manager_module):
    audio_manager = audio_manager_module(
        devices=[
            FakeDeviceInfo(name="CABLE Output (VB-Audio Virtual Cable)", maxInputChannels=2, defaultSampleRate=48000.0),
            FakeDeviceInfo(name="Speakers (Realtek(R) Audio)", maxOutputChannels=2, defaultSampleRate=48000.0),
        ]
    )

    cfg_path = tmp_path / "settings.cfg"
    cfg_path.write_text(
        "\n".join(
            [
                "[Devices]",
                "device_1 = Speakers (Realtek(R) Audio)",
                "",
                "[Settings]",
                "device_count = 1",
                "device_1_volume = 1.0",
                "device_1_latency = 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manager = audio_manager.AudioManager(config_path=str(cfg_path))

    # Simulate a transient/malformed reload where only Settings exists.
    import configparser

    manager.config = configparser.ConfigParser()
    manager.config["Settings"] = {"device_count": "1"}

    status = manager.get_status_string()
    assert "Devices:" in status
