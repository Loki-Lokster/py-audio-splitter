import os

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

def print_header():
    clear_screen()
    print(f"{Colors.CYAN}╔══════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.CYAN}║      Audio Splitter v1.0         ║{Colors.RESET}")
    print(f"{Colors.CYAN}╚══════════════════════════════════╝{Colors.RESET}\n")

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
    for i, device in enumerate(devices):
        print(f"{Colors.WHITE}{i + 1}. {device}{Colors.RESET}")

def get_user_choice(prompt, valid_options):
    while True:
        choice = input(f"{Colors.CYAN}{prompt}{Colors.RESET}").strip().lower()
        if choice in valid_options:
            return choice
        print(f"{Colors.RED}Invalid choice. Please try again.{Colors.RESET}") 