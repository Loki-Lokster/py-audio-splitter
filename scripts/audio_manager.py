import pyaudio
import numpy as np
import configparser
from typing import Dict, List, Optional
import time
import os
import threading
from .utils import (
    print_status, get_user_choice, Colors, 
    print_virtual_cable_info, create_windows_shortcut, 
    create_mac_shortcut, clear_screen, print_header
)
import platform
import logging
from .version import __version__
from dataclasses import dataclass

class AudioDevice:
    def __init__(self, name: str, device_index: int):
        self.name = name
        self.device_index = device_index
        self.volume = 1.0
        self.latency = 0.0
        self._stream = None
        self.sample_rate = 44100
        self.channels = 2
        self.buffer = np.zeros((self.sample_rate * 4, self.channels), dtype=np.float32)  # 4 seconds buffer
        self.buffer_position = 0
        self.is_buffer_full = False
        self.initial_offset = None  # Store initial offset from reference
        self.sync_check_interval = 5.0  # Check every 5 seconds
        self.last_sync_check = 0
        self.drift_tolerance = 0.02  # 20ms drift tolerance
        self.reset_count = 0
        self.max_resets = 3
        self.clock_initial_offset_seconds: Optional[float] = None
        self.clock_last_offset_seconds: Optional[float] = None
        self.clock_drift_seconds: Optional[float] = None
        self.last_drift_warning_time = 0.0
        self._drift_over_tolerance_since: Optional[float] = None

    def open_output_stream(
        self,
        pa: "pyaudio.PyAudio",
        *,
        audio_format: int,
        channels: int,
        rate: int,
        frames_per_buffer: int,
        callback=None,
    ):
        self.sample_rate = rate
        self.channels = channels
        self.buffer = np.zeros((self.sample_rate * 4, self.channels), dtype=np.float32)
        self.buffer_position = 0
        self.is_buffer_full = False
        self.initial_offset = None

        self._stream = pa.open(
            format=audio_format,
            channels=channels,
            rate=rate,
            output=True,
            input=False,
            output_device_index=self.device_index,
            frames_per_buffer=frames_per_buffer,
            stream_callback=callback,
        )
        
    def close_stream(self):
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            self._stream.close()
            self._stream = None

    def apply_volume(self, audio_data):
        return audio_data * self.volume

    def apply_latency(self, audio_data):
        if self.latency <= 0:
            return audio_data

        delay_samples = int(self.latency * self.sample_rate)
        data_len = len(audio_data)
        
        # Write new data to buffer
        next_pos = (self.buffer_position + data_len) % len(self.buffer)
        if next_pos < self.buffer_position:
            # Buffer wraps around
            first_part = len(self.buffer) - self.buffer_position
            self.buffer[self.buffer_position:] = audio_data[:first_part]
            self.buffer[:next_pos] = audio_data[first_part:]
        else:
            self.buffer[self.buffer_position:next_pos] = audio_data

        # Wait until we have enough data for the delay
        if not self.is_buffer_full:
            if self.buffer_position >= delay_samples:
                self.is_buffer_full = True
            else:
                self.buffer_position = next_pos
                return np.zeros_like(audio_data)

        # Read delayed data
        read_pos = (next_pos - delay_samples) % len(self.buffer)
        if read_pos + data_len <= len(self.buffer):
            delayed_audio = self.buffer[read_pos:read_pos + data_len].copy()
        else:
            # Handle wrap-around read
            first_part = len(self.buffer) - read_pos
            delayed_audio = np.concatenate((
                self.buffer[read_pos:],
                self.buffer[:data_len - first_part]
            ))

        self.buffer_position = next_pos
        return delayed_audio

    def reset_buffer(self):
        """Reset buffer and clear initial offset"""
        if self.reset_count > self.max_resets:
            return False

        self.buffer = np.zeros((self.sample_rate * 4, self.channels), dtype=np.float32)
        self.buffer_position = 0
        self.is_buffer_full = False
        self.initial_offset = None  # Clear initial offset to re-establish
        return True

    def check_sync(self, current_time, reference_device):
        """Check if device has drifted from its initial offset"""
        if reference_device is None:
            return True

        # Legacy placeholder: drift detection now uses stream clock drift in AudioManager.
        return True

    def process_audio(self, audio_data):
        return self.apply_volume(audio_data)


@dataclass
class _PlayoutState:
    read_pos: float
    rate_adjust: float = 0.0
    last_lag_error_seconds: float = 0.0

class AudioManager:
    def __init__(self, config_path='settings.cfg'):
        # Get the directory where the script is located
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Make config path absolute relative to script directory
        self.config_path = os.path.join(self.script_dir, config_path)
        
        self.config = configparser.ConfigParser()
        self._config_lock = threading.RLock()
        self.devices: Dict[str, AudioDevice] = {}
        self.pa = pyaudio.PyAudio()
        self.last_config_check = 0
        self.last_config_mtime = 0
        self.config_check_interval = 1.0  # Check every 1s
        self.max_devices = 8  # Set a reasonable maximum
        self.sync_check_interval = 2.0  # Check device sync every 2 seconds
        self.last_sync_check = 0
        self.initial_sync_established = False
        self.sync_warning_count = 0
        self.max_sync_warnings = 3
        self.current_log = ""  # Store current log message
        self._input_stream = None
        self._stop_event = threading.Event()
        self._audio_format = pyaudio.paFloat32
        self._format_preference = "auto"  # auto|float32|int16
        self._channels = 2
        self._sample_rate = 44100
        self._sample_rate_configured = False
        self._frames_per_buffer = 2048
        self._input_device_index = None
        self._input_device_name = ""
        self._virtual_cable_found = False
        self._last_drift_update = 0.0
        self._writer_thread = None
        self._buffer_lock = threading.Lock()
        self._ring_buffer = None
        self._ring_size_frames = 0
        self._write_pos = 0  # absolute frame counter
        self._base_buffer_seconds = 0.25
        self._max_rate_adjust = 0.001  # +/-1000 ppm
        self._kp = 0.02
        self._drift_log_interval = 15.0
        self._drift_persist_seconds = 2.0
        self._device_state: Dict[str, _PlayoutState] = {}
        self._reference_output_key: Optional[str] = None
        self.load_config()
        self.initialize_devices()

    def load_config(self):
        with self._config_lock:
            self.config.read(self.config_path, encoding='utf-8')
            if not self.config.has_section('Devices'):
                self._create_default_config()

    def _create_default_config(self):
        print_status("\nNo config found. Let's set up your audio devices.\n", "warning")
        devices = self.list_available_devices()
        
        if len(devices) < 2:
            print_status("Error: Need at least 2 audio output devices.", "error")
            return
        
        print_status("Available audio output devices:\n", "info")
        
        # Group devices by type with better formatting
        grouped_devices = {}
        device_indices = {}  # To store the mapping of display index to device name
        current_index = 1
        
        for device in devices:
            device_type = device.split('(')[0].strip()
            if device_type not in grouped_devices:
                grouped_devices[device_type] = []
            grouped_devices[device_type].append(device)
            device_indices[current_index] = device
            current_index += 1
        
        # Print grouped devices with clear separation
        for device_type in sorted(grouped_devices.keys()):
            print(f"\n{Colors.YELLOW}{device_type}:{Colors.RESET}")
            for device in grouped_devices[device_type]:
                idx = list(device_indices.keys())[list(device_indices.values()).index(device)]
                print(f"{Colors.WHITE}{idx:2d}. {device.strip()}{Colors.RESET}")
        
        try:
            # Get number of devices
            while True:
                device_count = input(f"\n{Colors.CYAN}How many output devices? (2-{self.max_devices}): {Colors.RESET}")
                if device_count.isdigit() and 2 <= int(device_count) <= self.max_devices:
                    device_count = int(device_count)
                    break
                print(f"{Colors.RED}Please enter a number between 2 and {self.max_devices}{Colors.RESET}")

            # Store selected devices and their settings
            selected_devices = {}
            device_volumes = {}
            selected_device_names = set()

            # Get devices and volumes
            for i in range(device_count):
                # Get device
                while True:
                    choice = input(f"\n{Colors.CYAN}Select device {i+1} (enter number): {Colors.RESET}")
                    if choice.isdigit() and 1 <= int(choice) < current_index:
                        device = device_indices[int(choice)]
                        if 'cable input' in device.lower():
                            confirm = get_user_choice(
                                f"\n{Colors.YELLOW}Warning:{Colors.RESET} '{device}' looks like a Virtual Cable playback device.\n"
                                "Selecting it as an output can create echo/feedback loops.\n"
                                "Use it anyway? (y/n): ",
                                {'y', 'n'}
                            )
                            if confirm != 'y':
                                continue
                        if device not in selected_device_names:
                            selected_device_names.add(device)
                            selected_devices[f'device_{i+1}'] = device
                            break
                        print(f"{Colors.RED}Device already selected. Please choose another.{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}Invalid choice. Please enter a number from the list.{Colors.RESET}")
                
                # Get volume
                while True:
                    vol = input(f"{Colors.CYAN}Enter volume for device {i+1} (0.0-1.0) [default: 1.0]: {Colors.RESET}").strip()
                    if not vol:
                        vol = "1.0"
                    try:
                        vol_float = float(vol)
                    except ValueError:
                        vol_float = None
                    if vol_float is not None and 0.0 <= vol_float <= 1.0:
                        device_volumes[f'device_{i+1}_volume'] = str(vol_float)
                        break
                    print(f"{Colors.RED}Invalid volume. Please enter a number between 0.0 and 1.0{Colors.RESET}")
            
            # Create the config
            self.config['Devices'] = selected_devices
            self.config['Settings'] = {
                'device_count': str(device_count),
                **device_volumes
            }
            
            # Add default latency settings
            for i in range(device_count):
                self.config['Settings'][f'device_{i+1}_latency'] = '0.0'

            # Stream settings (optional; safe defaults)
            self.config['Settings'].setdefault('input_device', 'auto')
            self.config['Settings'].setdefault('sample_rate', 'auto')
            self.config['Settings'].setdefault('format', 'auto')
            self.config['Settings'].setdefault('channels', '2')
            self.config['Settings'].setdefault('frames_per_buffer', '2048')

            # Drift compensation defaults
            self.config['Settings'].setdefault('base_buffer', '0.25')
            self.config['Settings'].setdefault('drift_kp', '0.02')
            self.config['Settings'].setdefault('drift_max_rate', '0.001')
            self.config['Settings'].setdefault('drift_log_interval', '15')
            self.config['Settings'].setdefault('drift_persist', '2')
            
            self.save_config()
            print_status("\nConfiguration saved! You can edit volumes and latency in settings.cfg", "success")
            
            # Ask about creating shortcut
            os_type = platform.system()
            if os_type in ['Windows', 'Darwin']:  # Darwin is macOS
                create_shortcut = get_user_choice(
                    "\nWould you like to create a desktop shortcut? (y/n): ",
                    {'y', 'n'}
                )
                
                if create_shortcut == 'y':
                    script_path = os.path.join(self.script_dir, 'audio_split.py')
                    success = False
                    
                    if os_type == 'Windows':
                        success = create_windows_shortcut(script_path)
                    else:  # macOS
                        success = create_mac_shortcut(script_path)
                    
                    if success:
                        print_status("Desktop shortcut created successfully!", "success")
            
            print_status("\nSetup complete! Starting application...", "success")
            
        except KeyboardInterrupt:
            print_status("\nSetup cancelled.", "warning")
            self.config['Devices'] = {}
            self.config['Settings'] = {}
            self.save_config()

    def initialize_devices(self):
        with self._config_lock:
            if not self.config.has_section('Devices') or not self.config['Devices']:
                return
        
        with self._config_lock:
            device_count = self.config.getint('Settings', 'device_count', fallback=2)
        logging.info(f"Initializing {device_count} devices")
        
        # Check for virtual cable
        virtual_cable_found = False
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if 'cable' in device_info['name'].lower():
                virtual_cable_found = True
                logging.info("Virtual Cable found")
                break
        self._virtual_cable_found = virtual_cable_found
        
        if not virtual_cable_found:
            logging.warning("Virtual Cable not found")
            print_virtual_cable_info()
        
        # Find and initialize all configured audio devices
        for i in range(1, device_count + 1):
            device_name = f'device_{i}'
            with self._config_lock:
                in_config = self.config.has_section('Devices') and device_name in self.config['Devices']
                device_friendly_name = self.config['Devices'][device_name] if in_config else None
            if device_friendly_name:
                if 'cable input' in device_friendly_name.lower():
                    logging.warning(
                        f"Configured output device '{device_friendly_name}' looks like Virtual Cable playback; "
                        "this can cause echo/feedback depending on Windows routing."
                    )
                device_index = self._find_device_index(device_friendly_name)
                
                if device_index is not None:
                    logging.info(f"Initializing device: {device_friendly_name} (index: {device_index})")
                    self.devices[device_name] = AudioDevice(device_friendly_name, device_index)
                    
                    # Apply initial settings
                    if self.config.has_section('Settings'):
                        volume = self.config.getfloat('Settings', f'{device_name}_volume', fallback=1.0)
                        latency = self.config.getfloat('Settings', f'{device_name}_latency', fallback=0.0)
                        self.update_device_settings(device_name, volume, latency, persist=False)
                        logging.info(f"Applied settings for {device_name}: volume={volume}, latency={latency}")
                else:
                    logging.error(f"Device not found: {device_friendly_name}")
                    print_status(f"Warning: Configured device '{device_friendly_name}' not found", "warning")

    def _find_device_index(self, device_name: str) -> Optional[int]:
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                # Make the device name matching more flexible
                if (device_name.lower() in device_info['name'].lower() or 
                    device_info['name'].lower() in device_name.lower()):
                    return i
        return None

    def audio_callback(self, in_data, frame_count, time_info, status):
        # Create silent audio data for testing
        audio_data = np.zeros((frame_count, 2), dtype=np.float32)
        
        # Process audio for the specific device that triggered the callback
        processed_data = audio_data.copy()
        
        # Return the processed data in the correct format for the callback
        return (processed_data.tobytes(), pyaudio.paContinue)

    def check_config_updated(self):
        """Check if config file has been modified"""
        current_time = time.time()
        
        # Only check periodically to avoid excessive file operations
        if current_time - self.last_config_check < self.config_check_interval:
            return False
            
        self.last_config_check = current_time
        
        try:
            mtime = os.path.getmtime(self.config_path)
            if mtime > self.last_config_mtime:
                self.last_config_mtime = mtime
                return True
        except OSError:
            pass  # File might not exist yet
        return False

    def reload_settings(self):
        """Reload settings from config file"""
        old_config = self.config
        new_config = configparser.ConfigParser()
        new_config.read(self.config_path, encoding='utf-8')
        
        if new_config.sections():
            with self._config_lock:
                self.config = new_config
            for device_name in self.devices:
                if new_config.has_section('Settings'):
                    try:
                        volume = new_config.getfloat('Settings', f'{device_name}_volume')
                        latency = new_config.getfloat('Settings', f'{device_name}_latency')
                        
                        if (volume != self.devices[device_name].volume or 
                            latency != self.devices[device_name].latency):
                            self.update_device_settings(device_name, volume, latency, persist=False)
                            logging.info(f"Updated settings for {device_name}: volume={volume}, latency={latency}")
                    except:
                        pass
            
            try:
                self.last_config_mtime = os.path.getmtime(self.config_path)
            except OSError:
                pass
        else:
            new_config = old_config
            with self._config_lock:
                self.config = new_config

    def should_process_audio(self):
        """Always process audio now that filtering is removed"""
        return True

    def start_audio(self):
        if not self.devices:
            raise RuntimeError("No output devices configured")

        self._load_stream_settings_from_config()
        self._load_drift_compensation_settings()

        input_device_index = self._find_input_device_index()
        self._input_device_index = input_device_index
        try:
            self._input_device_name = self.pa.get_device_info_by_index(input_device_index).get("name", "")
        except Exception:
            self._input_device_name = ""
        if not self._sample_rate_configured:
            try:
                info = self.pa.get_device_info_by_index(input_device_index)
                self._sample_rate = int(round(float(info.get("defaultSampleRate", self._sample_rate))))
            except Exception:
                pass
        logging.info(
            f"Starting audio streams: input_device_index={input_device_index}, "
            f"rate={self._sample_rate}, channels={self._channels}, frames_per_buffer={self._frames_per_buffer}"
        )

        self._stop_event.clear()

        # Best-effort: try float32 first, then int16 if the backend doesn't support float32.
        last_error = None
        if self._format_preference == "int16":
            format_order = (pyaudio.paInt16,)
        elif self._format_preference == "float32":
            format_order = (pyaudio.paFloat32,)
        else:
            format_order = (pyaudio.paFloat32, pyaudio.paInt16)

        for fmt in format_order:
            try:
                self._open_streams(input_device_index=input_device_index, audio_format=fmt)
                self._audio_format = fmt
                last_error = None
                break
            except Exception as e:
                last_error = e
                self._close_streams()

        if last_error is not None:
            raise last_error
        self._writer_thread = threading.Thread(target=self._writer_loop, name="audio-splitter-writer", daemon=True)
        self._writer_thread.start()

    def stop_audio(self, terminate_pa: bool = False):
        logging.info("Stopping audio streams")
        self._stop_event.set()
        self._close_streams()
        if self._writer_thread and self._writer_thread.is_alive():
            try:
                self._writer_thread.join(timeout=2.0)
            except Exception:
                pass
        self._writer_thread = None
        if terminate_pa:
            try:
                self.pa.terminate()
            except Exception:
                pass

    def shutdown(self):
        self.stop_audio(terminate_pa=True)

    def restart_audio(self):
        self.stop_audio(terminate_pa=False)
        self.start_audio()

    def get_configured_output_devices(self) -> List[str]:
        with self._config_lock:
            if not self.config.has_section("Devices"):
                return []
            device_count = self.config.getint("Settings", "device_count", fallback=0) if self.config.has_section("Settings") else 0
            out: List[str] = []
            for i in range(1, max(0, device_count) + 1):
                key = f"device_{i}"
                if key in self.config["Devices"]:
                    out.append(self.config["Devices"][key])
            return out

    def run_setup_wizard(self):
        self.stop_audio(terminate_pa=False)
        with self._config_lock:
            self.config = configparser.ConfigParser()
        self._create_default_config()
        self._reinitialize_devices_from_config()

    def set_output_devices(self, device_names: List[str], persist: bool = True):
        device_names = [str(n).strip() for n in device_names if str(n).strip()]
        if not device_names:
            raise ValueError("No output devices provided")

        with self._config_lock:
            old_devices = dict(self.config["Devices"]) if self.config.has_section("Devices") else {}
            old_settings = dict(self.config["Settings"]) if self.config.has_section("Settings") else {}

            if not self.config.has_section("Devices"):
                self.config.add_section("Devices")
            else:
                for k in list(self.config["Devices"].keys()):
                    self.config.remove_option("Devices", k)

            for i, name in enumerate(device_names, 1):
                self.config.set("Devices", f"device_{i}", name)

            if not self.config.has_section("Settings"):
                self.config.add_section("Settings")
            self.config.set("Settings", "device_count", str(len(device_names)))

            # Preserve volumes/latency by matching old configured names where possible.
            name_to_settings = {}
            for k, v in old_devices.items():
                if not k.startswith("device_"):
                    continue
                try:
                    idx = int(k.split("_", 1)[1])
                except Exception:
                    continue
                vol = old_settings.get(f"device_{idx}_volume", "1.0")
                lat = old_settings.get(f"device_{idx}_latency", "0.0")
                name_to_settings[v] = (vol, lat)

            for i, name in enumerate(device_names, 1):
                vol, lat = name_to_settings.get(name, ("1.0", "0.0"))
                self.config.set("Settings", f"device_{i}_volume", str(vol))
                self.config.set("Settings", f"device_{i}_latency", str(lat))

        if persist:
            self.save_config()
        self._reinitialize_devices_from_config()

    def _reinitialize_devices_from_config(self):
        self.devices = {}
        self.initialize_devices()

    def _open_streams(self, *, input_device_index: int, audio_format: int):
        self._input_stream = self.pa.open(
            format=audio_format,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            output=False,
            input_device_index=input_device_index,
            frames_per_buffer=self._frames_per_buffer,
        )

        self._setup_ring_buffer()
        self._setup_playout_states()

        for key, device in self.devices.items():
            device.open_output_stream(
                self.pa,
                audio_format=audio_format,
                channels=self._channels,
                rate=self._sample_rate,
                frames_per_buffer=self._frames_per_buffer,
                callback=self._make_output_callback(key),
            )
            try:
                device._stream.start_stream()
            except Exception:
                pass

    def _close_streams(self):
        if self._input_stream is not None:
            try:
                self._input_stream.stop_stream()
            except Exception:
                pass
            try:
                self._input_stream.close()
            except Exception:
                pass
            self._input_stream = None

        for device in self.devices.values():
            device.close_stream()
        with self._buffer_lock:
            self._ring_buffer = None
            self._ring_size_frames = 0
            self._write_pos = 0
        self._device_state = {}

    def _writer_loop(self):
        while not self._stop_event.is_set():
            if self.check_config_updated():
                self.reload_settings()
                self.update_log("Settings updated")

            try:
                in_data = self._input_stream.read(self._frames_per_buffer, exception_on_overflow=False)
            except Exception as e:
                self.update_log(f"Input read error: {e}")
                time.sleep(0.02)
                continue

            try:
                audio_data = self._decode_audio(in_data)
            except Exception as e:
                self.update_log(f"Decode error: {e}")
                continue

            with self._buffer_lock:
                if self._ring_buffer is None:
                    continue
                self._ring_write(audio_data)

    def _update_clock_drift(self, reference_device: AudioDevice, now: float):
        if now - self._last_drift_update < 0.25:
            return
        self._last_drift_update = now

        ref_stream = getattr(reference_device, "_stream", None)
        if ref_stream is None:
            return
        get_time = getattr(ref_stream, "get_time", None)
        if not callable(get_time):
            return

        try:
            ref_time = float(ref_stream.get_time())
        except Exception:
            return

        for device in self.devices.values():
            if device is reference_device:
                device.clock_last_offset_seconds = 0.0
                if device.clock_initial_offset_seconds is None:
                    device.clock_initial_offset_seconds = 0.0
                device.clock_drift_seconds = 0.0
                continue

            stream = getattr(device, "_stream", None)
            if stream is None:
                continue
            if not callable(getattr(stream, "get_time", None)):
                continue
            try:
                dev_time = float(stream.get_time())
            except Exception:
                continue

            offset = dev_time - ref_time
            device.clock_last_offset_seconds = offset
            if device.clock_initial_offset_seconds is None:
                device.clock_initial_offset_seconds = offset
            device.clock_drift_seconds = offset - float(device.clock_initial_offset_seconds)

            if device.clock_drift_seconds is not None and abs(device.clock_drift_seconds) > device.drift_tolerance:
                if now - device.last_drift_warning_time > 5.0:
                    device.last_drift_warning_time = now
                    self.update_log(f"Drift: {device.name} {device.clock_drift_seconds*1000.0:+.1f}ms")

    def _load_drift_compensation_settings(self):
        settings = self.config["Settings"] if self.config.has_section("Settings") else {}
        try:
            self._base_buffer_seconds = float(settings.get("base_buffer", self._base_buffer_seconds))
        except Exception:
            self._base_buffer_seconds = 0.25
        self._base_buffer_seconds = max(0.05, min(2.0, self._base_buffer_seconds))

        try:
            self._kp = float(settings.get("drift_kp", self._kp))
        except Exception:
            self._kp = 0.02
        self._kp = max(0.0, min(0.2, self._kp))

        try:
            self._max_rate_adjust = float(settings.get("drift_max_rate", self._max_rate_adjust))
        except Exception:
            self._max_rate_adjust = 0.001
        self._max_rate_adjust = max(0.0, min(0.01, self._max_rate_adjust))

        try:
            self._drift_log_interval = float(settings.get("drift_log_interval", self._drift_log_interval))
        except Exception:
            self._drift_log_interval = 15.0
        self._drift_log_interval = max(1.0, min(120.0, self._drift_log_interval))

        try:
            self._drift_persist_seconds = float(settings.get("drift_persist", self._drift_persist_seconds))
        except Exception:
            self._drift_persist_seconds = 2.0
        self._drift_persist_seconds = max(0.0, min(30.0, self._drift_persist_seconds))

    def _setup_ring_buffer(self):
        max_device_latency = 0.0
        for device in self.devices.values():
            max_device_latency = max(max_device_latency, float(device.latency))
        seconds = max(2.0, self._base_buffer_seconds + max_device_latency + 1.0)
        self._ring_size_frames = int(self._sample_rate * seconds)
        self._ring_buffer = np.zeros((self._ring_size_frames, self._channels), dtype=np.float32)
        self._write_pos = 0

    def _setup_playout_states(self):
        try:
            self._reference_output_key = min(self.devices.keys(), key=lambda k: self.devices[k].latency)
        except Exception:
            self._reference_output_key = next(iter(self.devices.keys()), None)

        base_delay_frames = int(self._base_buffer_seconds * self._sample_rate)
        self._device_state = {}
        for key, device in self.devices.items():
            target_delay = base_delay_frames + int(float(device.latency) * self._sample_rate)
            # Allow negative read positions so the initial target delay is met immediately via silent pre-roll.
            start_pos = float(self._write_pos - target_delay)
            self._device_state[key] = _PlayoutState(read_pos=start_pos)
            device.clock_initial_offset_seconds = None
            device.clock_last_offset_seconds = None
            device.clock_drift_seconds = None
            device._drift_over_tolerance_since = None

    def _ring_write(self, frames: np.ndarray):
        n = int(frames.shape[0])
        if n <= 0:
            return
        if self._ring_size_frames <= 0:
            return

        # If a write is larger than the ring, keep only the most recent window.
        # Advance write_pos so absolute indexing stays consistent.
        if n > self._ring_size_frames:
            drop = n - self._ring_size_frames
            self._write_pos += drop
            frames = frames[-self._ring_size_frames :, :]
            n = self._ring_size_frames

        start = int(self._write_pos % self._ring_size_frames)
        end = start + n
        if end <= self._ring_size_frames:
            self._ring_buffer[start:end, :] = frames
        else:
            first = self._ring_size_frames - start
            self._ring_buffer[start:, :] = frames[:first, :]
            self._ring_buffer[: end - self._ring_size_frames, :] = frames[first:, :]
        self._write_pos += n

    def _ring_read_linear(self, positions: np.ndarray) -> np.ndarray:
        out = np.zeros((int(positions.shape[0]), self._channels), dtype=np.float32)
        if self._ring_buffer is None or self._ring_size_frames <= 0:
            return out

        # Positions are absolute frame indices. For indices <0 or beyond what we've written,
        # return silence rather than wrapping around.
        max_pos = float(self._write_pos - 2)
        valid = (positions >= 0.0) & (positions <= max_pos)
        if not np.any(valid):
            return out

        pos_v = positions[valid]
        idx0 = np.floor(pos_v).astype(np.int64)
        frac = (pos_v - idx0.astype(np.float64)).astype(np.float32)
        idx1 = idx0 + 1

        i0 = (idx0 % self._ring_size_frames).astype(np.int64)
        i1 = (idx1 % self._ring_size_frames).astype(np.int64)

        a = self._ring_buffer[i0, :]
        b = self._ring_buffer[i1, :]
        frac2 = frac.reshape((-1, 1))
        out_v = (a * (1.0 - frac2)) + (b * frac2)
        out[valid, :] = out_v
        return out

    def _make_output_callback(self, device_key: str):
        def callback(in_data, frame_count, time_info, status):
            with self._buffer_lock:
                if self._ring_buffer is None or self._ring_size_frames <= 0:
                    return (np.zeros((frame_count, self._channels), dtype=np.float32).tobytes(), pyaudio.paContinue)

                state = self._device_state.get(device_key)
                device = self.devices.get(device_key)
                if state is None or device is None:
                    return (np.zeros((frame_count, self._channels), dtype=np.float32).tobytes(), pyaudio.paContinue)

                base_delay_frames = int(self._base_buffer_seconds * self._sample_rate)
                target_delay_frames = base_delay_frames + int(float(device.latency) * self._sample_rate)

                current_lag_frames = float(self._write_pos) - float(state.read_pos)
                lag_error_frames = current_lag_frames - float(target_delay_frames)
                lag_error_seconds = lag_error_frames / float(self._sample_rate)

                # Adaptive resampling: nudge read rate to keep lag near target.
                rate_adjust = self._kp * lag_error_seconds
                if rate_adjust > self._max_rate_adjust:
                    rate_adjust = self._max_rate_adjust
                elif rate_adjust < -self._max_rate_adjust:
                    rate_adjust = -self._max_rate_adjust
                state.rate_adjust = rate_adjust
                state.last_lag_error_seconds = lag_error_seconds
                step = 1.0 + float(rate_adjust)

                # Underflow/overrun protection: resync read pointer if too close or too far behind.
                min_needed = float(frame_count + 2)
                if current_lag_frames < min_needed or current_lag_frames > float(self._ring_size_frames - 4):
                    state.read_pos = float(self._write_pos - target_delay_frames)
                    device.clock_last_offset_seconds = current_lag_frames / float(self._sample_rate)
                    device.clock_drift_seconds = lag_error_seconds
                    out = np.zeros((frame_count, self._channels), dtype=np.float32)
                    return (self._encode_audio(out), pyaudio.paContinue)

                positions = state.read_pos + (np.arange(frame_count, dtype=np.float64) * step)
                out = self._ring_read_linear(positions)
                state.read_pos = float(state.read_pos + (frame_count * step))

                # Metrics: drift is "how far from target delay" (signed).
                device.clock_last_offset_seconds = current_lag_frames / float(self._sample_rate)
                if device.clock_drift_seconds is None:
                    device.clock_drift_seconds = lag_error_seconds
                else:
                    device.clock_drift_seconds = (0.8 * float(device.clock_drift_seconds)) + (0.2 * lag_error_seconds)
                if device.clock_initial_offset_seconds is None:
                    device.clock_initial_offset_seconds = device.clock_last_offset_seconds

                # Only log if drift persists, to avoid noisy startup/transients.
                drift_abs = abs(lag_error_seconds)
                if drift_abs > float(device.drift_tolerance):
                    if device._drift_over_tolerance_since is None:
                        device._drift_over_tolerance_since = time.time()
                    now = time.time()
                    if (
                        now - float(device._drift_over_tolerance_since) >= self._drift_persist_seconds
                        and now - device.last_drift_warning_time >= self._drift_log_interval
                    ):
                        device.last_drift_warning_time = now
                        self.update_log(
                            f"Drift: {device.name} {lag_error_seconds*1000.0:+.1f}ms "
                            f"(rate {rate_adjust*1_000_000.0:+.0f}ppm)"
                        )
                else:
                    device._drift_over_tolerance_since = None

                out = device.apply_volume(out)
                return (self._encode_audio(out), pyaudio.paContinue)

        return callback

    def _decode_audio(self, in_data: bytes) -> np.ndarray:
        if self._audio_format == pyaudio.paFloat32:
            audio = np.frombuffer(in_data, dtype=np.float32)
        elif self._audio_format == pyaudio.paInt16:
            audio_i16 = np.frombuffer(in_data, dtype=np.int16)
            audio = (audio_i16.astype(np.float32) / 32768.0)
        else:
            raise ValueError("Unsupported audio format")

        audio = audio.reshape((self._frames_per_buffer, self._channels))
        return audio

    def _encode_audio(self, audio: np.ndarray) -> bytes:
        if self._audio_format == pyaudio.paFloat32:
            return audio.astype(np.float32, copy=False).tobytes()
        if self._audio_format == pyaudio.paInt16:
            clipped = np.clip(audio, -1.0, 1.0)
            return (clipped * 32767.0).astype(np.int16).tobytes()
        raise ValueError("Unsupported audio format")

    def _find_input_device_index(self) -> int:
        configured = None
        if self.config.has_section("Settings"):
            configured = self.config.get("Settings", "input_device", fallback="").strip()
            if configured.lower() in ("", "auto", "default"):
                configured = None

        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if device_info.get("maxInputChannels", 0) <= 0:
                continue
            name = device_info.get("name", "")
            name_l = name.lower()
            if configured and (configured.lower() in name_l or name_l in configured.lower()):
                logging.info(f"Using configured input device: {name} (index: {i})")
                return i
            if not configured and "cable output" in name_l:
                logging.info(f"Using VB Cable input device: {name} (index: {i})")
                return i

        default_index = self.pa.get_default_input_device_info().get("index", 0)
        logging.info(f"Using default input device index: {default_index}")
        return int(default_index)

    def _load_stream_settings_from_config(self):
        settings = self.config["Settings"] if self.config.has_section("Settings") else {}

        try:
            self._channels = int(settings.get("channels", self._channels))
        except Exception:
            self._channels = 2

        try:
            self._frames_per_buffer = int(settings.get("frames_per_buffer", self._frames_per_buffer))
        except Exception:
            self._frames_per_buffer = 2048

        try:
            cfg_rate = settings.get("sample_rate", "").strip()
            if cfg_rate and cfg_rate.lower() not in ("auto", "default"):
                self._sample_rate = int(float(cfg_rate))
                self._sample_rate_configured = True
            else:
                self._sample_rate_configured = False
        except Exception:
            self._sample_rate_configured = False

        fmt = str(settings.get("format", "auto")).strip().lower()
        if fmt in ("auto", "default", ""):
            self._format_preference = "auto"
        elif fmt in ("float32", "f32"):
            self._format_preference = "float32"
        elif fmt in ("int16", "i16"):
            self._format_preference = "int16"
        else:
            self._format_preference = "auto"

    def set_stream_setting(self, key: str, value: str, persist: bool = True):
        key = str(key).strip()
        value = str(value).strip()
        if not key:
            raise ValueError("Missing key")

        with self._config_lock:
            if not self.config.has_section("Settings"):
                self.config.add_section("Settings")
            self.config.set("Settings", key, value)

        if persist:
            self.save_config()

        # Keep in-memory settings in sync for restart.
        if key in ("sample_rate", "channels", "frames_per_buffer"):
            self._load_stream_settings_from_config()

    def get_runtime_info(self) -> Dict[str, str]:
        fmt = "float32" if self._audio_format == pyaudio.paFloat32 else "int16" if self._audio_format == pyaudio.paInt16 else str(self._audio_format)
        return {
            "version": __version__,
            "config_path": self.config_path,
            "virtual_cable": "yes" if self._virtual_cable_found else "no",
            "input_device_name": self._input_device_name,
            "sample_rate": str(self._sample_rate),
            "channels": str(self._channels),
            "frames_per_buffer": str(self._frames_per_buffer),
            "format": fmt,
            "log": self.current_log or "",
        }

    def get_output_device_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for key, device in self.devices.items():
            if device.clock_drift_seconds is None:
                drift_ms = "-"
            else:
                drift_ms = f"{device.clock_drift_seconds * 1000.0:+.0f}"
            rate_ppm = "-"
            state = self._device_state.get(key)
            if state is not None:
                rate_ppm = f"{state.rate_adjust * 1_000_000.0:+.0f}"
            rows.append(
                {
                    "key": key,
                    "name": device.name,
                    "volume": f"{device.volume:.2f}",
                    "latency_ms": f"{device.latency * 1000.0:.0f}",
                    "drift_ms": drift_ms,
                    "rate_ppm": rate_ppm,
                }
            )
        return rows

    def update_device_settings(
        self,
        device_name: str,
        volume: Optional[float] = None,
        latency: Optional[float] = None,
        persist: bool = True,
    ):
        if device_name not in self.devices:
            raise ValueError(f"Unknown device: {device_name}")
            
        device = self.devices[device_name]
        
        if volume is not None:
            device.volume = max(0.0, min(1.0, volume))
            if persist:
                with self._config_lock:
                    if not self.config.has_section('Settings'):
                        self.config.add_section('Settings')
                    self.config.set('Settings', f'{device_name}_volume', str(volume))
            
        if latency is not None:
            device.latency = max(0.0, latency)
            if persist:
                with self._config_lock:
                    if not self.config.has_section('Settings'):
                        self.config.add_section('Settings')
                    self.config.set('Settings', f'{device_name}_latency', str(latency))
            
        if persist:
            self.save_config()

    def save_config(self):
        with self._config_lock:
            with open(self.config_path, 'w', encoding='utf-8', newline='\n') as configfile:
                self.config.write(configfile)

    @staticmethod
    def list_available_devices():
        """List all available audio output devices"""
        pa = pyaudio.PyAudio()
        devices = {}  # Use dict to track full names
        
        for i in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                name = device_info['name']
                
                # Skip "Microsoft Sound Mapper" and "Primary Sound Driver"
                if any(skip in name for skip in ["Microsoft Sound Mapper", "Primary Sound Driver"]):
                    continue
                    
                # Check if this is a truncated version of an existing name
                is_truncated = False
                for full_name in list(devices.keys()):
                    if name in full_name or full_name in name:
                        if len(full_name) > len(name):
                            is_truncated = True
                        else:
                            # Remove shorter version if we find a longer one
                            devices.pop(full_name)
                            devices[name] = i
                            break
                
                # Add new name if it's not truncated
                if not is_truncated:
                    devices[name] = i
        
        pa.terminate()
        
        # Sort by name but group similar devices together
        def sort_key(name):
            # Extract base name without parentheses
            base = name.split('(')[0].strip()
            return (base, name)
        
        return sorted(devices.keys(), key=sort_key)

    @staticmethod
    def list_available_input_devices():
        """List all available audio input devices"""
        pa = pyaudio.PyAudio()
        devices = {}

        for i in range(pa.get_device_count()):
            device_info = pa.get_device_info_by_index(i)
            if device_info.get("maxInputChannels", 0) > 0:
                name = device_info.get("name", "")
                if any(skip in name for skip in ["Microsoft Sound Mapper", "Primary Sound Driver"]):
                    continue
                devices[name] = i

        pa.terminate()

        def sort_key(name):
            base = name.split("(")[0].strip()
            return (base, name)

        return sorted(devices.keys(), key=sort_key)

    def get_status_string(self):
        """Get current status string for display"""
        status_lines = []
        
        # Add header with controls
        status_lines.append(f"{Colors.WHITE}Controls:{Colors.RESET}")
        status_lines.append(f"  {Colors.CYAN}Ctrl+C{Colors.RESET} to stop application")
        status_lines.append(f"  Edit {Colors.YELLOW}settings.cfg{Colors.RESET} to adjust volumes/latency")
        status_lines.append("")  # Empty line for spacing
        
        # Add device statuses
        status_lines.append(f"{Colors.WHITE}Devices:{Colors.RESET}")
        for name, device in self.devices.items():
            with self._config_lock:
                if self.config.has_section('Devices') and name in self.config['Devices']:
                    device_name = self.config['Devices'][name]
                else:
                    device_name = device.name
            if len(device_name) > 30:
                device_name = device_name[:27] + "..."
            
            status_lines.append(
                f"  {Colors.CYAN}{device_name}{Colors.RESET}:"
                f"\n    Volume: {Colors.GREEN}{device.volume:.2f}{Colors.RESET}"
                f"\n    Latency: {Colors.YELLOW}{device.latency*1000:.0f}ms{Colors.RESET}"
                + (
                    f"\n    Drift: {Colors.RED}{device.clock_drift_seconds*1000:+.0f}ms{Colors.RESET}"
                    if device.clock_drift_seconds is not None
                    else ""
                )
            )
        
        # Add current log message if any
        if self.current_log:
            status_lines.append("")
            status_lines.append(f"{Colors.WHITE}Log:\n  {Colors.RED}{self.current_log}{Colors.RESET}")
        
        return "\n".join(status_lines)

    def update_log(self, message):
        """Update the current log message"""
        self.current_log = message

    def reset_config(self):
        """Reset configuration file"""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        with self._config_lock:
            self.config = configparser.ConfigParser()
        self._create_default_config()

    def check_devices_sync(self, current_time):
        """Check sync between all devices"""
        if not self.initial_sync_established:
            # Initialize sync references for all devices
            for device in self.devices.values():
                device.initial_offset = None
            self.initial_sync_established = True
            return True

        if current_time - self.last_sync_check < self.sync_check_interval:
            return True

        self.last_sync_check = current_time

        # Get reference device (one with lowest latency)
        reference_device = min(self.devices.values(), key=lambda d: d.latency)
        
        for device in self.devices.values():
            if device == reference_device:
                continue

            # Calculate current offset from reference
            current_offset = (device.buffer_position - reference_device.buffer_position) / float(reference_device.sample_rate)

            # Check for drift from initial offset
            drift = abs(current_offset - device.initial_offset)
            
            if drift > device.drift_tolerance:
                self.sync_warning_count += 1
                if self.sync_warning_count >= self.max_sync_warnings:
                    logging.warning(f"Persistent sync mismatch: {device.name} is off by {drift*1000:.1f}ms")
                    self.sync_warning_count = 0
                    return False
            else:
                self.sync_warning_count = max(0, self.sync_warning_count - 1)

        return True

def main():
    clear_screen()
    print_header()
    print("Available audio devices:")
    devices = AudioManager.list_available_devices()
    for i, device in enumerate(devices):
        print(f"{i + 1}. {device}")
        
    print("\nMake sure your settings.cfg matches these device names!")
    
    manager = AudioManager()
    
    # Only proceed if devices are configured
    if not manager.devices:
        return
    
    try:
        manager.start_audio()
        print_status("Audio streaming started. Press Ctrl+C to stop\n", "success")
        
        last_status = ""
        while True:
            # Update status line with current settings
            status = manager.get_status_string()
            if status != last_status:
                clear_screen()
                print_header()
                print_status(status, "info")
                last_status = status
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print_status("\nShutting down...", "warning")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 
