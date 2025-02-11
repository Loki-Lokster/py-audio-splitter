import argparse
from scripts.utils import print_header, print_status, Colors, clear_screen
from scripts.audio_manager import AudioManager
import time
import signal
import sys

def signal_handler(sig, frame):
    print_status("\nShutting down...", "warning")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Audio Splitter - Route audio to multiple output devices')
    parser.add_argument('--config', type=str, default='settings.cfg',
                       help='Path to config file (default: settings.cfg)')
    parser.add_argument('--list-devices', action='store_true',
                       help='List available audio devices and exit')
    parser.add_argument('--reset-config', action='store_true',
                       help='Reset configuration file')
    
    args = parser.parse_args()
    
    # Setup signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    print_header()
    
    manager = AudioManager(args.config)
    
    if args.list_devices:
        devices = AudioManager.list_available_devices()
        print_status("Available audio output devices:\n", "info")
        
        # Group devices by type
        grouped_devices = {}
        for device in devices:
            device_type = device.split('(')[0].strip()
            if device_type not in grouped_devices:
                grouped_devices[device_type] = []
            grouped_devices[device_type].append(device)
        
        # Print grouped devices
        current_index = 1
        for device_type in sorted(grouped_devices.keys()):
            print(f"\n{Colors.YELLOW}{device_type}:{Colors.RESET}")
            for device in grouped_devices[device_type]:
                print(f"{Colors.WHITE}{current_index:2d}. {device}{Colors.RESET}")
                current_index += 1
        return
        
    if args.reset_config:
        print_status("Resetting configuration...", "warning")
        manager.reset_config()
        print_status("Configuration reset. Please restart the application.", "success")
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
    except Exception as e:
        print_status(f"\nError: {str(e)}", "error")
    finally:
        manager.stop_audio()

if __name__ == "__main__":
    main() 