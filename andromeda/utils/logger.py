from typing import Any
from colorama import Fore, Style, init
from pyeztrace.tracer import Logging

# Initialize colorama
init(autoreset=True)

# Custom log functions with color


def log_supervisor(message: str):
    Logging.log_info(f"{Fore.BLUE}[SUPERVISOR] {message}{Style.RESET_ALL}")


def log_agent(agent_name: str, message: str):
    Logging.log_info(f"{Fore.BLACK}[AGENT: {agent_name}] {message}{Style.RESET_ALL}")


def log_tool(tool_name: str, message: str):
    Logging.log_info(f"{Fore.YELLOW}[TOOL: {tool_name}] {message}{Style.RESET_ALL}")


def log_input(message: str):
    Logging.log_info(f"{Fore.CYAN}[INPUT] {message}{Style.RESET_ALL}")


def log_output(message: Any, pretty: bool = False):
    if pretty:
        if isinstance(message, dict):
            for key, value in message.items():
                Logging.log_info(
                    f"{Fore.MAGENTA}[OUTPUT] {key}: {value}{Style.RESET_ALL}"
                )
        elif isinstance(message, list):
            for item in message:
                Logging.log_info(f"{Fore.MAGENTA}[OUTPUT] {item}{Style.RESET_ALL}")
        else:
            Logging.log_info(f"{Fore.MAGENTA}[OUTPUT] {message}{Style.RESET_ALL}")
    else:
        Logging.log_info(f"{Fore.MAGENTA}[OUTPUT] {message}{Style.RESET_ALL}")


def log_error(message: str, *args, **kwargs):
    Logging.log_error(f"{Fore.RED}[ERROR] {message}{Style.RESET_ALL}", *args, **kwargs)


def log_warning(message: str, *args, **kwargs):
    Logging.log_warning(f"{Fore.YELLOW}[WARNING] {message}{Style.RESET_ALL}", *args, **kwargs)