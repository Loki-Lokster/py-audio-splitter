import os
import sys
import platform
import subprocess

# ANSI color codes
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(version="1.0.0"):
    """Print a stylized ASCII art header"""
    clear_screen()
    
    ascii_art = f"""
    {Colors.WHITE}╔═════════════════════════════════╗{Colors.CYAN}
    {Colors.WHITE}║  {Colors.YELLOW}A U D I O {Colors.WHITE}// {Colors.CYAN}S P L I T T E R{Colors.WHITE}   ║{Colors.CYAN}
    {Colors.WHITE}╚═════════════════════════════════╝{Colors.CYAN}
    {Colors.RED}{" "*13}v {version}{Colors.WHITE}
    {Colors.WHITE}{" "*12}╼━━━━━━━━╾{Colors.RESET}
    """
    print(ascii_art)

def print_status(message, status_type="info"):
    """Print status message with color coding"""
    colors = {
        "info": Colors.BLUE,
        "success": Colors.GREEN,
        "error": Colors.RED,
        "warning": Colors.YELLOW
    }
    color = colors.get(status_type, Colors.WHITE)
    print(f"\r{color}{message}{Colors.RESET}", end='')

def print_devices(devices):
    print(f"{Colors.YELLOW}Available audio devices:{Colors.RESET}")
    for i, device in enumerate(devices, 1):
        # Ensure each device is on its own line with proper padding
        print(f"{Colors.WHITE}{i:2d}. {device.strip()}{Colors.RESET}")

def get_user_choice(prompt, valid_options):
    while True:
        choice = input(f"{Colors.CYAN}{prompt}{Colors.RESET}").strip().lower()
        if choice in valid_options:
            return choice
        print(f"{Colors.RED}Invalid choice. Please try again.{Colors.RESET}")

def print_virtual_cable_info():
    """Print information about installing Virtual Cable"""
    print(f"\n{Colors.YELLOW}Virtual Audio Cable not detected!{Colors.RESET}")
    print(f"{Colors.WHITE}To route audio between applications, you need to install Virtual Audio Cable:")
    print(f"{Colors.CYAN}https://vb-audio.com/Cable/{Colors.RESET}")
    print(f"{Colors.WHITE}After installation, restart this application.{Colors.RESET}\n")

def create_windows_shortcut(script_path):
    """Create Windows desktop shortcut using a simple .bat file"""
    try:
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        bat_path = os.path.join(desktop, "Audio Splitter.bat")
        
        # Create batch file
        with open(bat_path, 'w') as f:
            f.write('@echo off\n')
            f.write(f'"{sys.executable}" "{script_path}"\n')
            f.write('pause')
        
        return True
    except Exception as e:
        print_status(f"Failed to create shortcut: {str(e)}", "error")
        return False

def create_mac_shortcut(script_path):
    """Create macOS desktop shortcut"""
    try:
        desktop = os.path.expanduser("~/Desktop")
        path = os.path.join(desktop, "Audio Splitter.command")
        
        with open(path, 'w') as f:
            f.write('#!/bin/bash\n')
            f.write(f'"{sys.executable}" "{script_path}"')
        
        # Make the file executable
        os.chmod(path, 0o755)
        return True
    except Exception as e:
        print_status(f"Failed to create shortcut: {str(e)}", "error")
        return False

def setup_console_window(setup_mode=False):
    """Setup console window size and properties"""
    if platform.system() == 'Windows':
        try:
            # Use wider window during setup
            if setup_mode:
                os.system('mode con: cols=80 lines=30')
            else:
                os.system('mode con: cols=40 lines=25')
            # Set window title
            os.system('title Audio Splitter')
        except:
            pass  # Fail silently if window manipulation fails 

