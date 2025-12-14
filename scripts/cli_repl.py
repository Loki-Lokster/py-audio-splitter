import shlex
import time
import shutil
import re
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from .utils import Colors, clear_screen


_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _display_width(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


@dataclass
class CommandResult:
    message: str = ""
    should_exit: bool = False
    should_render: bool = True


class CliRepl:
    def __init__(self, manager):
        self.manager = manager
        self._last_message = ""
        self._title_color = random.choice(
            [
                Colors.CYAN,
                Colors.GREEN,
                Colors.YELLOW,
                Colors.BLUE,
                Colors.WHITE,
            ]
        )

    def run(self):
        self._render()
        while True:
            try:
                line = input(f"{Colors.CYAN}audio-splitter{Colors.RESET} {_DIM}›{Colors.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not line:
                self._render()
                continue

            result = self._dispatch(line)
            if result.message:
                self._last_message = result.message
                if not result.should_render:
                    print(result.message)
            if result.should_exit:
                return
            if result.should_render:
                self._render()

    def _dispatch(self, line: str) -> CommandResult:
        try:
            parts = shlex.split(line)
        except ValueError as e:
            return CommandResult(message=f"Parse error: {e}")

        if not parts:
            return CommandResult()

        cmd = parts[0].lower()
        args = parts[1:]

        handlers: Dict[str, Callable[[List[str]], CommandResult]] = {
            "?": self._cmd_help,
            "help": self._cmd_help,
            "status": self._cmd_status,
            "clear": self._cmd_clear,
            "list": self._cmd_list,
            "outputs": self._cmd_outputs,
            "add": self._cmd_add,
            "remove": self._cmd_remove,
            "set": self._cmd_set,
            "reload": self._cmd_reload,
            "restart": self._cmd_restart,
            "watch": self._cmd_watch,
            "wizard": self._cmd_wizard,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler is None:
            return CommandResult(message=f"Unknown command: {cmd}. Try `help`.")
        return handler(args)

    def _cmd_help(self, _args: List[str]) -> CommandResult:
        msg = "\n".join(
            [
                "Commands (quick start):",
                "  1) `list outputs`         show all available output devices",
                "  2) `outputs`              show currently selected outputs",
                "  3) `add output <i>`        add output #i from `list outputs` (restarts)",
                "     `add output <text>`     add output by name match (restarts)",
                "  4) `remove output <N>`     remove configured device_N (restarts)",
                "     `remove output <text>`  remove by name match (restarts)",
                "",
                "General:",
                "  help | ?                  Show this help",
                "  status                    Re-render dashboard",
                "  clear                     Clear screen",
                "  watch                     Print status every second (Ctrl+C to stop)",
                "  quit | exit               Quit",
                "",
                "Device lists:",
                "  list outputs              List available output devices",
                "  list inputs               List available input devices",
                "  outputs                   Show configured output devices",
                "",
                "Edit outputs (auto-restarts):",
                "  add output <i>             Add an output by index from `list outputs`",
                "  add output <text>          Add an output by name match",
                "  remove output <N>          Remove configured device_N",
                "  remove output <text>       Remove a configured output by name match",
                "  set output <N> <i>         Replace configured device_N with output #i",
                "  set outputs <i> <j> ...    Replace the entire outputs list at once",
                "  wizard                    Re-run the interactive setup wizard",
                "",
                "Tuning:",
                "  set volume <N> <0..1>      Set device_N volume",
                "  set latency <N> <seconds>  Set device_N latency",
                "  set input <substring>      Set input device (restarts)",
                "  set sample_rate <hz>       Set sample rate (restarts)",
                "  set channels <n>           Set channels (restarts)",
                "  set buffer <frames>        Set frames_per_buffer (restarts)",
                "  reload                    Reload volumes/latency from settings.cfg",
                "  restart                   Restart audio streams",
            ]
        )
        return CommandResult(message=msg)

    def _cmd_status(self, _args: List[str]) -> CommandResult:
        return CommandResult()

    def _cmd_clear(self, _args: List[str]) -> CommandResult:
        clear_screen()
        return CommandResult(should_render=True)

    def _cmd_list(self, args: List[str]) -> CommandResult:
        if not args:
            return CommandResult(message="Usage: list outputs|inputs")
        which = args[0].lower()
        if which in ("outputs", "output"):
            outputs = self.manager.list_available_devices()
            lines = ["Available output devices:"]
            lines += [f"  {i:2d}. {name}" for i, name in enumerate(outputs, 1)]
            return CommandResult(message="\n".join(lines), should_render=False)
        if which in ("inputs", "input"):
            inputs = self.manager.list_available_input_devices()
            lines = ["Available input devices:"]
            lines += [f"  {i:2d}. {name}" for i, name in enumerate(inputs, 1)]
            return CommandResult(message="\n".join(lines), should_render=False)
        return CommandResult(message="Usage: list outputs|inputs")

    def _cmd_outputs(self, _args: List[str]) -> CommandResult:
        configured = self.manager.get_configured_output_devices()
        if not configured:
            return CommandResult(message="No configured outputs. Run `wizard` or `add output <i>`.", should_render=False)
        lines = ["Configured output devices:"]
        for i, name in enumerate(configured, 1):
            lines.append(f"  device_{i}: {name}")
        lines.append("")
        lines.append("Tip: `list outputs` then `add output <i>` or `set output <N> <i>`.")
        return CommandResult(message="\n".join(lines), should_render=False)

    def _cmd_add(self, args: List[str]) -> CommandResult:
        if len(args) < 2 or args[0].lower() != "output":
            return CommandResult(message="Usage: add output <i>  (use `list outputs` indices)")
        outputs = self.manager.list_available_devices()

        name = None
        if len(args) == 2 and args[1].isdigit():
            idx = int(args[1])
            if idx < 1 or idx > len(outputs):
                return CommandResult(message=f"Invalid output index: {idx}")
            name = outputs[idx - 1]
        else:
            query = " ".join(args[1:]).strip()
            if not query:
                return CommandResult(message="Usage: add output <i|text>")
            matches = [d for d in outputs if query.lower() in d.lower()]
            if not matches:
                return CommandResult(message="No matching output device found. Use `list outputs`.", should_render=False)
            if len(matches) > 1:
                lines = ["Multiple outputs match. Be more specific or use `add output <i>`:", ""]
                for i, d in enumerate(outputs, 1):
                    if d in matches:
                        lines.append(f"  {i:2d}. {d}")
                return CommandResult(message="\n".join(lines), should_render=False)
            name = matches[0]

        configured = self.manager.get_configured_output_devices()
        if name in configured:
            return CommandResult(message="That output is already configured.", should_render=False)
        configured.append(name)
        self.manager.set_output_devices(configured, persist=True)
        self.manager.restart_audio()
        return CommandResult(message=f"Added output: {name} (restarted)", should_render=False)

    def _cmd_remove(self, args: List[str]) -> CommandResult:
        if len(args) < 2 or args[0].lower() != "output":
            return CommandResult(message="Usage: remove output <N>  (removes device_N)")
        configured = self.manager.get_configured_output_devices()
        if len(configured) <= 1:
            return CommandResult(message="Refusing to remove the last output device. Use `wizard` to reconfigure.", should_render=False)
        removed = None
        if len(args) == 2 and args[1].isdigit():
            n = int(args[1])
            if n < 1 or n > len(configured):
                return CommandResult(message=f"Invalid configured device: device_{n}")
            removed = configured.pop(n - 1)
        else:
            query = " ".join(args[1:]).strip()
            if not query:
                return CommandResult(message="Usage: remove output <N|text>", should_render=False)
            matches = [(i, name) for i, name in enumerate(configured, 1) if query.lower() in name.lower()]
            if not matches:
                return CommandResult(message="No configured output matches that text. Use `outputs`.", should_render=False)
            if len(matches) > 1:
                lines = ["Multiple configured outputs match. Be more specific or use `remove output <N>`:", ""]
                for i, name in matches:
                    lines.append(f"  device_{i}: {name}")
                return CommandResult(message="\n".join(lines), should_render=False)
            i, name = matches[0]
            removed = configured.pop(i - 1)
        self.manager.set_output_devices(configured, persist=True)
        self.manager.restart_audio()
        return CommandResult(message=f"Removed output: {removed} (restarted)", should_render=False)

    def _cmd_set(self, args: List[str]) -> CommandResult:
        if len(args) < 2:
            return CommandResult(message="Usage: set <key> ... (try `help`)")
        key = args[0].lower()
        rest = args[1:]

        if key == "output":
            if len(rest) != 2:
                return CommandResult(message="Usage: set output <N> <i>  (replace device_N with output #i)")
            try:
                n = int(rest[0])
                idx = int(rest[1])
            except ValueError:
                return CommandResult(message="Usage: set output <N> <i>  (replace device_N with output #i)")
            outputs = self.manager.list_available_devices()
            if idx < 1 or idx > len(outputs):
                return CommandResult(message=f"Invalid output index: {idx}")
            configured = self.manager.get_configured_output_devices()
            if n < 1 or n > len(configured):
                return CommandResult(message=f"Invalid configured device: device_{n}")
            name = outputs[idx - 1]
            if name in configured and configured[n - 1] != name:
                return CommandResult(message="That output is already configured.", should_render=False)
            configured[n - 1] = name
            self.manager.set_output_devices(configured, persist=True)
            self.manager.restart_audio()
            return CommandResult(message=f"Set device_{n} to: {name} (restarted)", should_render=False)

        if key == "outputs":
            outputs = self.manager.list_available_devices()
            indices = []
            try:
                indices = [int(x) for x in rest]
            except ValueError:
                return CommandResult(message="Usage: set outputs <i> <j> ... (use `list outputs` indices)")
            if not indices:
                return CommandResult(message="Usage: set outputs <i> <j> ... (use `list outputs` indices)")
            for idx in indices:
                if idx < 1 or idx > len(outputs):
                    return CommandResult(message=f"Invalid output index: {idx}")
            device_names = [outputs[i - 1] for i in indices]
            self.manager.set_output_devices(device_names, persist=True)
            self.manager.restart_audio()
            return CommandResult(message=f"Selected {len(device_names)} output device(s) and restarted streams", should_render=False)

        if key in ("volume", "latency"):
            if len(rest) != 2:
                return CommandResult(message=f"Usage: set {key} <N> <value>")
            try:
                n = int(rest[0])
            except ValueError:
                return CommandResult(message="Device index must be an integer (e.g., 1)")
            device_name = f"device_{n}"
            try:
                value = float(rest[1])
            except ValueError:
                return CommandResult(message="Value must be a number")

            if key == "volume":
                self.manager.update_device_settings(device_name, volume=value, persist=True)
                return CommandResult(message=f"Updated {device_name} volume to {value:g}")
            self.manager.update_device_settings(device_name, latency=value, persist=True)
            return CommandResult(message=f"Updated {device_name} latency to {value:g}s")

        if key in ("input", "sample_rate", "channels", "buffer", "frames_per_buffer"):
            if not rest:
                return CommandResult(message=f"Usage: set {key} <value>")
            value_str = " ".join(rest).strip()
            config_key = key
            if key == "input":
                config_key = "input_device"
            if key == "buffer":
                config_key = "frames_per_buffer"
            self.manager.set_stream_setting(config_key, value_str, persist=True)
            self.manager.restart_audio()
            return CommandResult(message=f"Applied {config_key}={value_str} and restarted streams")

        return CommandResult(message=f"Unknown setting: {key}")

    def _cmd_reload(self, _args: List[str]) -> CommandResult:
        self.manager.reload_settings()
        for device in self.manager.devices.values():
            device.reset_buffer()
        return CommandResult(message="Reloaded settings.cfg (volume/latency)")

    def _cmd_restart(self, _args: List[str]) -> CommandResult:
        self.manager.restart_audio()
        return CommandResult(message="Restarted audio streams")

    def _cmd_watch(self, _args: List[str]) -> CommandResult:
        try:
            while True:
                clear_screen()
                print(self._render_dashboard())
                time.sleep(1.0)
        except KeyboardInterrupt:
            return CommandResult(message="Stopped watch mode")

    def _cmd_wizard(self, _args: List[str]) -> CommandResult:
        self.manager.run_setup_wizard()
        self.manager.restart_audio()
        return CommandResult(message="Setup wizard complete; restarted streams")

    def _cmd_quit(self, _args: List[str]) -> CommandResult:
        return CommandResult(should_exit=True, should_render=False)

    def _render(self):
        clear_screen()
        print(self._render_dashboard())
        if self._last_message:
            print()
            print(f"{Colors.YELLOW}{_BOLD}Last:{Colors.RESET} {self._last_message}")

    def _render_dashboard(self) -> str:
        info = self.manager.get_runtime_info()
        outputs = self.manager.get_output_device_rows()

        width = max(60, min(100, shutil.get_terminal_size(fallback=(80, 24)).columns))
        title_text = f" {self._title_color}Audio Splitter{Colors.RESET} v{info.get('version','')} "
        pad = max(0, width - 2 - _display_width(title_text))
        left_pad = pad // 2
        right_pad = pad - left_pad
        banner = [
            f"{Colors.CYAN}╭{'─' * (width - 2)}╮{Colors.RESET}",
            f"{Colors.CYAN}│{Colors.RESET}{_BOLD}{Colors.WHITE}{(' ' * left_pad) + title_text + (' ' * right_pad)}{Colors.RESET}{Colors.CYAN}│{Colors.RESET}",
            f"{Colors.CYAN}╰{'─' * (width - 2)}╯{Colors.RESET}",
        ]

        title = "\n".join(banner)
        top = [
            title,
            f"{_DIM}{info.get('config_path','')}{Colors.RESET}",
            "",
            f"{Colors.WHITE}{_BOLD}Input{Colors.RESET}",
            f"  Device: {Colors.CYAN}{info.get('input_device_name','(not started)')}{Colors.RESET}",
            f"  Rate:   {Colors.CYAN}{info.get('sample_rate','')}{Colors.RESET}  "
            f"Ch: {Colors.CYAN}{info.get('channels','')}{Colors.RESET}  "
            f"Buf: {Colors.CYAN}{info.get('frames_per_buffer','')}{Colors.RESET}  "
            f"Fmt: {Colors.CYAN}{info.get('format','')}{Colors.RESET}",
            "",
            f"{Colors.WHITE}{_BOLD}Outputs{Colors.RESET}",
        ]

        rows = []
        for row in outputs:
            drift_text = f"{row['drift_ms']}ms"
            drift_color = Colors.GREEN
            try:
                drift_val = float(row["drift_ms"])
                if abs(drift_val) >= 20:
                    drift_color = Colors.RED
                elif abs(drift_val) >= 10:
                    drift_color = Colors.YELLOW
            except Exception:
                drift_color = Colors.WHITE
            rows.append(
                f"  {Colors.CYAN}{row['key']}{Colors.RESET}  "
                f"{row['name']}  "
                f"{_DIM}vol{Colors.RESET} {Colors.GREEN}{row['volume']}{Colors.RESET}  "
                f"{_DIM}lat{Colors.RESET} {Colors.YELLOW}{row['latency_ms']}ms{Colors.RESET}  "
                f"{_DIM}drift{Colors.RESET} {drift_color}{drift_text}{Colors.RESET}  "
                f"{_DIM}rate{Colors.RESET} {Colors.CYAN}{row.get('rate_ppm','-')}ppm{Colors.RESET}"
            )

        bottom = [
            "",
            f"{Colors.WHITE}{_BOLD}Log{Colors.RESET}",
            f"  {Colors.RED}{info.get('log','')}{Colors.RESET}" if info.get("log") else f"  {_DIM}(none){Colors.RESET}",
            "",
            f"{_DIM}Type `help` for commands. In watch mode, Ctrl+C returns to the REPL.{Colors.RESET}",
        ]

        return "\n".join(top + rows + bottom)
