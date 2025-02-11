import os
from colorama import init, Fore, Style

# Initialize colorama for Windows support
init()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    clear_screen()
    print(f"{Fore.CYAN}╔══════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║      Audio Splitter v1.0         ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚══════════════════════════════════╝{Style.RESET_ALL}\n")

def print_status(message, status_type="info"):
    """Print status message with color coding"""
    colors = {
        "info": Fore.BLUE,
        "success": Fore.GREEN,
        "error": Fore.RED,
        "warning": Fore.YELLOW
    }
    color = colors.get(status_type, Fore.WHITE)
    print(f"\r{color}{message}{Style.RESET_ALL}", end='')

def print_devices(devices):
    print(f"{Fore.YELLOW}Available audio devices:{Style.RESET_ALL}")
    for i, device in enumerate(devices):
        print(f"{Fore.WHITE}{i + 1}. {device}{Style.RESET_ALL}")

def get_user_choice(prompt, valid_options):
    while True:
        choice = input(f"{Fore.CYAN}{prompt}{Style.RESET_ALL}").strip().lower()
        if choice in valid_options:
            return choice
        print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}") 