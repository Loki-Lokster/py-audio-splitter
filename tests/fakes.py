import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FakeDeviceInfo:
    name: str
    maxOutputChannels: int = 0
    maxInputChannels: int = 0
    defaultSampleRate: float = 44100.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "maxOutputChannels": self.maxOutputChannels,
            "maxInputChannels": self.maxInputChannels,
            "defaultSampleRate": self.defaultSampleRate,
        }


class FakeStream:
    def __init__(self, *, rate: int, channels: int, format: int, stream_callback=None):
        self.rate = rate
        self.channels = channels
        self.format = format
        self.stream_callback = stream_callback
        self.started = False
        self.closed = False
        self.writes: List[bytes] = []
        self._time = 0.0

    def start_stream(self):
        self.started = True

    def stop_stream(self):
        self.started = False

    def close(self):
        self.closed = True

    def read(self, frame_count: int, exception_on_overflow: bool = False) -> bytes:
        self._time += float(frame_count) / float(self.rate)
        if self.format == 1:  # paFloat32
            data = np.zeros((frame_count, self.channels), dtype=np.float32)
            return data.tobytes()
        if self.format == 2:  # paInt16
            data = np.zeros((frame_count, self.channels), dtype=np.int16)
            return data.tobytes()
        raise ValueError("Unsupported fake format")

    def write(self, data: bytes):
        self.writes.append(data)
        bytes_per_sample = 4 if self.format == 1 else 2
        frame_count = len(data) // (bytes_per_sample * self.channels)
        self._time += float(frame_count) / float(self.rate)

    def get_time(self) -> float:
        return float(self._time)


class FakePyAudio:
    def __init__(self, devices: List[FakeDeviceInfo]):
        self._devices = devices

    def get_device_count(self) -> int:
        return len(self._devices)

    def get_device_info_by_index(self, i: int) -> Dict[str, Any]:
        return self._devices[i].as_dict()

    def get_default_input_device_info(self) -> Dict[str, Any]:
        for idx, device in enumerate(self._devices):
            if device.maxInputChannels > 0:
                info = device.as_dict()
                info["index"] = idx
                return info
        return {"index": 0}

    def open(
        self,
        *,
        format: int,
        channels: int,
        rate: int,
        output: bool = False,
        input: bool = False,
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        frames_per_buffer: int = 1024,
        stream_callback=None,
    ):
        return FakeStream(rate=rate, channels=channels, format=format, stream_callback=stream_callback)

    def terminate(self):
        return None


def make_pyaudio_module(devices: List[FakeDeviceInfo]):
    fake_module = types.ModuleType("pyaudio")
    fake_module.paFloat32 = 1
    fake_module.paInt16 = 2
    fake_module.paContinue = 0
    fake_module.PyAudio = lambda: FakePyAudio(devices)
    return fake_module
