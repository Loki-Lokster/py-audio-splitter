import pyaudio
import numpy as np
import configparser
from typing import Dict, Optional
import time
import os
from .utils import print_status, get_user_choice, Colors, print_virtual_cable_info, create_windows_shortcut, create_mac_shortcut
import platform
import logging

class AudioDevice:
    def __init__(self, name: str, device_index: int):
        self.name = name
        self.device_index = device_index
        self.volume = 1.0
        self.latency = 0.0
        self._stream = None
        self._pa = pyaudio.PyAudio()
        # Increase buffer size to handle larger delays
        self.buffer = np.zeros((88200, 2), dtype=np.float32)  # 2 seconds buffer
        self.buffer_position = 0
        self.is_buffer_full = False

    def open_stream(self, callback):
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=44100,
            output=True,
            input=True,
            input_device_index=self._find_input_device(),
            output_device_index=self.device_index,
            frames_per_buffer=1024,
            stream_callback=callback
        )
        self._stream.start_stream()
        
    def close_stream(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._pa.terminate()

    def apply_volume(self, audio_data):
        return audio_data * self.volume

    def apply_latency(self, audio_data):
        if self.latency <= 0:
            return audio_data

        delay_samples = int(self.latency * 44100)
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

    def process_audio(self, audio_data):
        # First apply latency (if any)
        processed = self.apply_latency(audio_data.copy())
        # Then apply volume
        return self.apply_volume(processed)

    def _find_input_device(self):
        # Try to find CABLE Output as input device
        for i in range(self._pa.get_device_count()):
            device_info = self._pa.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0 and 'CABLE Output' in device_info['name']:
                return i
        # Fallback to default input device
        return self._pa.get_default_input_device_info()['index']

class AudioManager:
    def __init__(self, config_path='settings.cfg'):
        # Get the directory where the script is located
        self.script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Make config path absolute relative to script directory
        self.config_path = os.path.join(self.script_dir, config_path)
        
        self.config = configparser.ConfigParser()
        self.devices: Dict[str, AudioDevice] = {}
        self.pa = pyaudio.PyAudio()
        self.last_config_check = 0
        self.last_config_mtime = 0
        self.config_check_interval = 1.0  # Check every 1s
        self.max_devices = 8  # Set a reasonable maximum
        self.load_config()
        self.initialize_devices()

    def load_config(self):
        self.config.read(self.config_path)
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

            # Get devices and volumes
            for i in range(device_count):
                # Get device
                while True:
                    choice = input(f"\n{Colors.CYAN}Select device {i+1} (enter number): {Colors.RESET}")
                    if choice.isdigit() and 1 <= int(choice) < current_index:
                        if choice not in selected_devices.values():
                            device = device_indices[int(choice)]
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
                    if vol.replace(".", "").isdigit() and 0 <= float(vol) <= 1:
                        device_volumes[f'device_{i+1}_volume'] = vol
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
        if not self.config['Devices']:
            return
        
        device_count = self.config.getint('Settings', 'device_count', fallback=2)
        logging.info(f"Initializing {device_count} devices")
        
        # Check for virtual cable
        virtual_cable_found = False
        for i in range(self.pa.get_device_count()):
            device_info = self.pa.get_device_info_by_index(i)
            if 'CABLE' in device_info['name']:
                virtual_cable_found = True
                logging.info("Virtual Cable found")
                break
        
        if not virtual_cable_found:
            logging.warning("Virtual Cable not found")
            print_virtual_cable_info()
        
        # Find and initialize all configured audio devices
        for i in range(1, device_count + 1):
            device_name = f'device_{i}'
            if device_name in self.config['Devices']:
                device_friendly_name = self.config['Devices'][device_name]
                device_index = self._find_device_index(device_friendly_name)
                
                if device_index is not None:
                    logging.info(f"Initializing device: {device_friendly_name} (index: {device_index})")
                    self.devices[device_name] = AudioDevice(device_friendly_name, device_index)
                    
                    # Apply initial settings
                    if self.config.has_section('Settings'):
                        volume = self.config.getfloat('Settings', f'{device_name}_volume', fallback=1.0)
                        latency = self.config.getfloat('Settings', f'{device_name}_latency', fallback=0.0)
                        self.update_device_settings(device_name, volume, latency)
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
        """Reload settings from config file without reinitializing devices"""
        old_config = self.config
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        
        if self.config.sections():
            settings_changed = False
            for device_name in self.devices:
                if self.config.has_section('Settings'):
                    try:
                        volume = self.config.getfloat('Settings', f'{device_name}_volume')
                        latency = self.config.getfloat('Settings', f'{device_name}_latency')
                        
                        if (volume != self.devices[device_name].volume or 
                            latency != self.devices[device_name].latency):
                            settings_changed = True
                            self.update_device_settings(device_name, volume, latency)
                    except:
                        pass
            
            try:
                self.last_config_mtime = os.path.getmtime(self.config_path)
            except OSError:
                pass
        else:
            self.config = old_config

    def should_process_audio(self):
        """Always process audio now that filtering is removed"""
        return True

    def start_audio(self):
        logging.info("Starting audio streams")
        def create_device_callback(device):
            def callback(in_data, frame_count, time_info, status):
                # Check for config updates
                if self.check_config_updated():
                    self.reload_settings()
                    device.is_buffer_full = False
                    device.buffer_position = 0
                
                # Convert input data to numpy array
                if in_data:  # Always process
                    audio_data = np.frombuffer(in_data, dtype=np.float32)
                    audio_data = audio_data.reshape((frame_count, 2))
                    
                    # Process audio using the device's processing chain
                    processed_data = device.process_audio(audio_data)
                    
                    return (processed_data.tobytes(), pyaudio.paContinue)
                return (np.zeros(frame_count * 2, dtype=np.float32).tobytes(), pyaudio.paContinue)
            return callback

        # Store initial config modification time
        try:
            self.last_config_mtime = os.path.getmtime(self.config_path)
        except OSError:
            self.last_config_mtime = 0

        # Open streams with device-specific callbacks
        for device in self.devices.values():
            device.open_stream(create_device_callback(device))

    def stop_audio(self):
        logging.info("Stopping audio streams")
        for device in self.devices.values():
            device.close_stream()

    def update_device_settings(self, device_name: str, volume: Optional[float] = None, 
                             latency: Optional[float] = None):
        if device_name not in self.devices:
            raise ValueError(f"Unknown device: {device_name}")
            
        device = self.devices[device_name]
        
        if volume is not None:
            device.volume = max(0.0, min(1.0, volume))
            self.config.set('Settings', f'{device_name}_volume', str(volume))
            
        if latency is not None:
            device.latency = max(0.0, latency)
            self.config.set('Settings', f'{device_name}_latency', str(latency))
            
        self.save_config()

    def save_config(self):
        with open(self.config_path, 'w') as configfile:
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
            device_name = self.config['Devices'][name]
            # Truncate device name if too long
            if len(device_name) > 30:
                device_name = device_name[:27] + "..."
            
            status_lines.append(
                f"  {Colors.CYAN}{device_name}{Colors.RESET}:"
                f"\n    Volume: {Colors.GREEN}{device.volume:.2f}{Colors.RESET}"
                f"\n    Latency: {Colors.YELLOW}{device.latency*1000:.0f}ms{Colors.RESET}"
            )
        
        return "\n".join(status_lines)

    def reset_config(self):
        """Reset configuration file"""
        if os.path.exists(self.config_path):
            os.remove(self.config_path)
        self.config = configparser.ConfigParser()
        self._create_default_config()

def main():
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