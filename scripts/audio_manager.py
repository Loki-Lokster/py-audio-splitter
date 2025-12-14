import pyaudio
import numpy as np
import configparser
from typing import Dict, Optional
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

    def open_output_stream(self, pa: "pyaudio.PyAudio", *, audio_format: int, channels: int, rate: int, frames_per_buffer: int):
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

        # Only check periodically
        if current_time - self.last_sync_check < self.sync_check_interval:
            return True

        self.last_sync_check = current_time

        # Calculate current offset from reference
        current_offset = (self.buffer_position - reference_device.buffer_position) / float(self.sample_rate)

        # On first check, store the initial offset
        if self.initial_offset is None:
            self.initial_offset = current_offset
            return True

        # Check for drift from initial offset
        drift = abs(current_offset - self.initial_offset)
        
        if drift > self.drift_tolerance:
            self.reset_count += 1
            if self.reset_count <= self.max_resets:
                return False
        else:
            self.reset_count = max(0, self.reset_count - 1)

        return True

    def process_audio(self, audio_data):
        current_time = time.time()
        
        # Check sync status
        if not self.check_sync(current_time, None):
            if self.reset_buffer():
                self.initial_offset = None
            return np.zeros_like(audio_data)
        
        # First apply latency (if any)
        processed = self.apply_latency(audio_data.copy())
        # Then apply volume
        return self.apply_volume(processed)

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
        self._audio_thread = None
        self._stop_event = threading.Event()
        self._audio_format = pyaudio.paFloat32
        self._channels = 2
        self._sample_rate = 44100
        self._sample_rate_configured = False
        self._frames_per_buffer = 2048
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

        input_device_index = self._find_input_device_index()
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
        for fmt in (pyaudio.paFloat32, pyaudio.paInt16):
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

        self._audio_thread = threading.Thread(target=self._audio_loop, name="audio-splitter-loop", daemon=True)
        self._audio_thread.start()

    def stop_audio(self):
        logging.info("Stopping audio streams")
        self._stop_event.set()
        if self._audio_thread and self._audio_thread.is_alive():
            try:
                self._audio_thread.join(timeout=2.0)
            except Exception:
                pass
        self._audio_thread = None
        self._close_streams()
        try:
            self.pa.terminate()
        except Exception:
            pass

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

        for device in self.devices.values():
            device.open_output_stream(
                self.pa,
                audio_format=audio_format,
                channels=self._channels,
                rate=self._sample_rate,
                frames_per_buffer=self._frames_per_buffer,
            )

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

    def _audio_loop(self):
        reference_device = None
        try:
            reference_device = min(self.devices.values(), key=lambda d: d.latency)
        except ValueError:
            reference_device = None

        while not self._stop_event.is_set():
            if self.check_config_updated():
                self.reload_settings()
                for device in self.devices.values():
                    device.reset_buffer()
                self.update_log("Settings updated")

            try:
                in_data = self._input_stream.read(self._frames_per_buffer, exception_on_overflow=False)
            except Exception as e:
                self.update_log(f"Input read error: {e}")
                time.sleep(0.05)
                continue

            try:
                audio_data = self._decode_audio(in_data)
            except Exception as e:
                self.update_log(f"Decode error: {e}")
                continue

            current_time = time.time()
            for device in self.devices.values():
                if reference_device is not None and device is not reference_device:
                    if not device.check_sync(current_time, reference_device):
                        if device.reset_buffer():
                            self.update_log(f"Resetting {device.name} - drift detected")
                            continue
                try:
                    processed = device.process_audio(audio_data)
                    out_bytes = self._encode_audio(processed)
                    if device._stream is not None:
                        device._stream.write(out_bytes)
                except Exception as e:
                    self.update_log(f"Output error for {device.name}: {e}")

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
            if cfg_rate:
                self._sample_rate = int(float(cfg_rate))
                self._sample_rate_configured = True
            else:
                self._sample_rate_configured = False
        except Exception:
            self._sample_rate_configured = False

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
