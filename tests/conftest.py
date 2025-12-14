import importlib
import sys

import pytest

from tests.fakes import FakeDeviceInfo, make_pyaudio_module
from typing import List


@pytest.fixture
def audio_manager_module(monkeypatch):
    def _load(devices: List[FakeDeviceInfo]):
        monkeypatch.setitem(sys.modules, "pyaudio", make_pyaudio_module(devices))
        import scripts.audio_manager as audio_manager

        return importlib.reload(audio_manager)

    return _load
