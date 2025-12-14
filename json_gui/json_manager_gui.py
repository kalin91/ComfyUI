"""GUI Application for managing JSON configuration files."""

import importlib
import inspect
import json
import os
import logging
import random
import re
from pathlib import Path
from typing import Any, Callable, Optional, cast
import threading
import uuid
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import yaml
from PIL import Image, ImageTk
import torch
from app.logger import setup_logger
import json_gui.utils as gui_utils
from comfy.cli_args import args
from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES, SCHEDULER_HANDLERS
from custom_nodes.ComfyUI_Impact_Pack.modules.impact.core import ADDITIONAL_SCHEDULERS
from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import FaceDetailer

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

COMBO_CONSTANTS = {
    "sampler_names": SAMPLER_NAMES,
    "scheduler_names": SCHEDULER_NAMES,
    "scheduler_handlers": list(SCHEDULER_HANDLERS) + ADDITIONAL_SCHEDULERS,
    "sam_detection_hint": FaceDetailer.INPUT_TYPES()["required"]["sam_detection_hint"][0],
}


def _create_string_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    notify_change: Callable[[], None],
    string_entries: dict[str, tk.Entry],
) -> None:
    """Create a string entry widget."""
    entry = ttk.Entry(frame, width=60)
    entry.insert(0, str(value))
    entry.bind("<KeyRelease>", lambda e: notify_change())
    entry.pack(side="left", padx=5, fill="x", expand=True)
    assert isinstance(value, str), f"Value for key '{key}' must be a string"
    string_entries[full_key] = entry


def _create_boolean_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    notify_change: Callable[[], None],
    boolean_vars: dict[str, tk.BooleanVar],
) -> None:
    """Create a boolean entry widget."""
    assert isinstance(value, bool), f"Value for key '{key}' must be a bool in list item '{key}'"
    label = ttk.Label(frame, text=f"{key}:", width=25, anchor="e")
    label.pack(side="left")
    var = tk.BooleanVar(value=value)
    check = ttk.Checkbutton(frame, variable=var, command=notify_change)
    check.pack(side="left", padx=5)
    boolean_vars[full_key] = var


def _create_open_preview_handler(p_combo: ttk.Combobox, p_folder: str, p_frame: ttk.Widget) -> Callable[[], None]:
    """Create a handler to open a preview window."""

    def _open_preview(folder=p_folder, combo=p_combo, frame=p_frame) -> None:
        """Open a floating preview window for the selected image."""
        path = os.path.join(folder, combo.get())
        if not path:
            messagebox.showwarning("Preview", "Select a file to preview")
            return
        try:
            img = Image.open(path)
        except Exception as e:
            messagebox.showerror("Preview Error", f"Cannot open image:\n{e}")
            return

        parent_win = frame.winfo_toplevel()
        win = tk.Toplevel(parent_win)
        win.title(f"Preview - {os.path.basename(path)}")
        win.transient(parent_win)
        win.resizable(True, True)

        # Compute max preview size relative to screen
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        max_w = min(int(sw * 0.7), 1600)
        max_h = min(int(sh * 0.8), 1200)
        try:
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        except Exception:
            img.thumbnail((max_w, max_h))

        photo = ImageTk.PhotoImage(img)
        img_label = ttk.Label(win, image=photo)
        img_label.image = photo  # Keep reference
        img_label.pack(padx=12, pady=12)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=(0, 12))

    return _open_preview


def _create_file_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    body: dict[str, Any],
    notify_change: Callable[[], None],
    file_entries: dict[str, ttk.Combobox],
) -> None:
    """Create a file entry widget."""
    assert isinstance(value, str), f"Value for key '{key}' must be a string"
    assert "parent" in body[key], f"'parent' not specified for file type key '{key}' in body"
    body_parent = body[key]["parent"]
    assert isinstance(body_parent, str), f"'parent' for file type key '{key}' must be a string"
    label = ttk.Label(frame, text=f"{key}:", width=25, anchor="e")
    label.pack(side="left")
    combo = ttk.Combobox(frame, width=57, state="readonly")
    files: list[str]
    folder: str
    combo.bind("<<ComboboxSelected>>", lambda e: notify_change())
    combo.pack(side="left", padx=5, fill="x", expand=True)
    if body_parent == "input":
        files, folder = gui_utils.get_input_files_recursive()
        ttk.Button(frame, text="Preview", command=_create_open_preview_handler(combo, folder, frame)).pack(
            side="left", padx=(5, 5)
        )
    else:
        files, folder = gui_utils.get_folder_files_recursive(body_parent)
    combo["values"] = files
    if value in files:
        combo.set(value)
    file_entries[full_key] = combo


def _create_combo_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    body: dict[str, Any],
    notify_change: Callable[[], None],
    combo_entries: dict[str, ttk.Combobox],
) -> None:
    """Create a combo entry widget."""
    assert isinstance(value, str), f"Value for key '{key}' must be a string"
    assert (
        "constant" in body[key] or "values" in body[key]
    ), f"'constant' or 'values' not specified for combo type key '{key}' in body"
    combo_values: list[str]
    if "values" in body[key]:
        combo_values = body[key]["values"]
    else:
        assert body[key]["constant"] in COMBO_CONSTANTS, (
            f"Constant '{body[key]['constant']}' not found " f"in constants dictionary for combo type key '{key}'"
        )
        combo_values = COMBO_CONSTANTS[body[key]["constant"]]
    label = ttk.Label(frame, text=f"{key}:", width=25, anchor="e")
    label.pack(side="left")
    combo = ttk.Combobox(frame, width=57, state="readonly")
    combo["values"] = combo_values
    if value in combo_values:
        combo.set(value)
    combo.bind("<<ComboboxSelected>>", lambda e: notify_change())
    combo.pack(side="left", padx=5, fill="x", expand=True)
    combo_entries[full_key] = combo


def _create_multiline_text_widget(
    parent: ttk.Widget,
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    indent: int,
    on_text_modified: Callable[[tk.Event], None],
    text_entries: dict[str, tk.Text],
) -> None:
    """Create a multiline text widget."""
    assert isinstance(value, str), f"Value for key '{key}' must be a string"
    # Multiline text box for positive/negative prompts
    label = ttk.Label(frame, text=f"{key}:", font=("TkDefaultFont", 10, "bold"))
    label.pack(anchor="w")

    text_frame = ttk.Frame(parent)
    text_frame.pack(fill="x", padx=(indent * 20 + 10, 5), pady=5)

    text_widget = tk.Text(text_frame, height=8, width=80, wrap="word")
    text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
    text_widget.configure(yscrollcommand=text_scrollbar.set)

    text_widget.insert("1.0", str(value))
    text_widget.bind("<<Modified>>", on_text_modified)

    text_widget.pack(side="left", fill="both", expand=True)
    text_scrollbar.pack(side="right", fill="y")

    text_entries[full_key] = text_widget


def _create_number_validator(
    min_val: float, max_val: float, type_val: type, last_val: tuple[list[Any], list[ttk.Spinbox]], format_str: str
) -> Callable[[str], bool]:
    """ "Create a number validator function."""

    def validate_number(
        value: str,
        p_min=min_val,
        p_max=max_val,
        t_val: type = type_val,
        l_val: tuple[list[Any], list[ttk.Spinbox]] = last_val,
        p_format: str = format_str,
    ) -> bool:
        """Validate number input."""
        try:
            last_value = l_val[0][0] if l_val[0] else None
            assert last_value or last_value == 0, "Last value not found for validation"
            entry_widget: ttk.Spinbox = l_val[1][0] if l_val[1] else None
            assert entry_widget, "Entry widget not found for validation"

            def reset_value() -> None:
                """Reset entry to last valid value."""
                entry_widget.set(last_value)
                entry_widget.config(validate="focusout")

            if value in ("", "-", "."):
                value = "0"

            val = t_val(value)
            if val < p_min or val > p_max:
                entry_widget.after_idle(reset_value)
                return False
            if t_val == float and "." in value:
                formatted_val = p_format % val
                max_decimals: int = len(formatted_val.split(".")[-1])
                actual_decimals: int = len(value.split(".")[-1])
                if actual_decimals > max_decimals:
                    entry_widget.after_idle(reset_value)
                    return False
            l_val[0][0] = val
            entry_widget.after_idle(lambda: entry_widget.config(validate="focusout"))
            return True
        except ValueError:
            entry_widget.after_idle(reset_value)
            return False
        except Exception:
            logging.exception("Unexpected error during validation")
            entry_widget.after_idle(reset_value)
            return False

    return validate_number


def _create_on_invalid_handler(
    body_type: str, min_val: float, max_val: float, format_str: str, notify_change: Callable[[], None]
) -> Callable[[], None]:
    """Create an invalid input handler."""

    def on_invalid(
        b_type=body_type, p_min=min_val, p_max=max_val, p_format=format_str, on_change=notify_change
    ) -> None:
        """Handle invalid input."""
        max_decimals = 0 if b_type == "int" else int(p_format.replace("f", "").split(".")[-1])
        messagebox.showwarning(
            "Invalid Input",
            (
                f"Please enter a valid {b_type} between {p_min} and "
                f"{p_max} with up to {max_decimals} decimal places."
            ),
        )
        on_change()

    return on_invalid


def _create_randomize_handler(
    entry: ttk.Spinbox,
    min_val: float,
    max_val: float,
    format_str: str,
    body_type: str,
    notify_change: Callable[[], None],
) -> Callable[[], None]:
    """Create a randomize button handler."""

    def set_random(e=entry, mn=min_val, mx=max_val, fmt=format_str, bt=body_type, on_change=notify_change) -> None:
        """Set a random value in the entry."""
        on_change()
        if bt == "int":
            val = random.randint(int(mn), int(mx))
            e.set(val)
        else:
            val = random.uniform(float(mn), float(mx))
            e.set(fmt % val)

    return set_random


def _create_numeric_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    register_call: Callable[[Callable[..., Any]], str],
    notify_change: Callable[[], None],
    body: dict[str, Any],
    body_type: str,
    full_key: str,
    int_entries: dict[str, tk.Entry],
    float_entries: dict[str, tk.Entry],
) -> None:
    """Create a numeric entry widget."""
    type_val: type = int if body_type == "int" else float
    min_val: float = body[key].get("min", -999999999999999)
    max_val: float = body[key].get("max", 999999999999999)
    format_str: str = "%.0f" if body_type == "int" else body[key].get("format", "%.1f")
    last_val: tuple[list[Any], list[ttk.Spinbox]] = ([value], [])

    entry = ttk.Spinbox(
        frame,
        from_=min_val,
        to=max_val,
        increment=body[key].get("step", 1.0),
        width=25,
        wrap=True,
        format=format_str,
        command=notify_change,
        validate="focusout",
        validatecommand=(
            register_call(_create_number_validator(min_val, max_val, type_val, last_val, format_str)),
            "%P",
        ),
        invalidcommand=(
            register_call(_create_on_invalid_handler(body_type, min_val, max_val, format_str, notify_change)),
        ),
    )
    entry.set(type_val(value))
    last_val[1].append(entry)
    entry.bind("<KeyRelease>", lambda e: notify_change())
    entry.pack(side="left", padx=(0, 5))
    if body[key].get("randomizable", False):
        entry.config(foreground="blue")
        ttk.Button(
            frame,
            text="Random",
            command=_create_randomize_handler(entry, min_val, max_val, format_str, body_type, notify_change),
        ).pack(side="left", padx=(0, 5))

        assert isinstance(type_val(value), type_val), f"Value for key '{key}' must be an {body_type}"
    if body_type == "int":
        int_entries[full_key] = entry
    elif body_type == "float":
        float_entries[full_key] = entry


def _show_loading_modal(parent, message="Loading...") -> tk.Toplevel:
    """Show a modal loading window."""
    loading_win = tk.Toplevel(parent)
    loading_win.title("")
    loading_win.geometry("250x80")
    loading_win.transient(parent)
    loading_win.grab_set()  # Hace la ventana modal
    loading_win.resizable(False, False)
    loading_win.protocol("WM_DELETE_WINDOW", lambda: None)  # Deshabilita cerrar

    # Frame for text and scrollbar
    frame = ttk.Frame(loading_win)
    frame.pack(fill="both", expand=True, padx=10, pady=10)

    text_widget = tk.Text(frame, height=3, wrap="word")
    text_widget.insert("1.0", message)
    text_widget.config(state="disabled")
    text_widget.pack(side="left", fill="both", expand=True)

    # Horizontal scrollbar
    x_scroll = ttk.Scrollbar(frame, orient="horizontal", command=text_widget.xview)
    text_widget.configure(xscrollcommand=x_scroll.set)
    x_scroll.pack(side="bottom", fill="x")
    return loading_win


class JSONTreeEditor(ttk.Frame):
    """A hierarchical, editable view for JSON data."""

    def __init__(self, parent: tk.Widget, on_change: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.data: dict[str, Any] = {}
        self.string_entries: dict[str, tk.Entry] = {}
        self.int_entries: dict[str, tk.Entry] = {}
        self.float_entries: dict[str, tk.Entry] = {}
        self.text_entries: dict[str, tk.Text] = {}  # For multiline text widgets
        self.boolean_vars: dict[str, tk.BooleanVar] = {}
        self.list_entries: dict[str, list[dict[str, Any]]] = {}
        self.file_entries: dict[str, ttk.Combobox] = {}
        self.combo_entries: dict[str, ttk.Combobox] = {}
        self._on_change = on_change  # Callback when any value changes

        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_data(self, data: dict[str, Any], body: dict[str, Any]) -> None:
        """Load JSON data into the editor."""
        self.data = data
        self.string_entries.clear()
        self.int_entries.clear()
        self.float_entries.clear()
        self.text_entries.clear()
        self.boolean_vars.clear()
        self.list_entries.clear()
        self.file_entries.clear()
        self.combo_entries.clear()

        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        assert "props" in body, "'props' key not found in body"

        self._build_tree(self.scrollable_frame, data, body["props"], "")
        # Use after_idle to ensure all widget events are processed before marking as clean
        self.after_idle(lambda: self._on_change(False))

    def _build_tree(
        self, parent: tk.Widget, data: dict[str, Any], body: dict[str, Any], prefix: str, indent: int = 0
    ) -> None:
        """Recursively build the tree view."""
        # List of keys that should use multiline text boxes
        try:
            for key, value in data.items():
                assert key in body, f"'{key}' key not found in body"
                assert isinstance(body[key], dict), f"Body for key '{key}' must be a dict"
                assert "type" in body[key], f"'type' not specified for key '{key}' in body"
                body_type: str = body[key]["type"]
                assert "isArray" in body[key], f"'isArray' not specified for key '{key}' in body"
                body_is_array: bool = body[key]["isArray"]
                full_key = f"{prefix}.{key}" if prefix else key
                frame = ttk.Frame(parent)
                frame.pack(fill="x", padx=(indent * 20, 5), pady=2)

                if body_is_array:
                    assert isinstance(value, list), f"Value for key '{key}' must be a list"
                    body[key]["isArray"] = False  # Temporarily set to False to process items
                    label = ttk.Label(frame, text=f"▼ {key} (Array):", font=("TkDefaultFont", 10, "bold"))
                    label.pack(anchor="w")
                    for i, item in enumerate(value):
                        item_key = f"{full_key}[{i}]"
                        item_frame = ttk.Frame(parent)
                        item_frame.pack(fill="x", padx=((indent + 1) * 20, 5), pady=2)
                        item_label = ttk.Label(item_frame, text=f"{key} [{i}]:", font=("TkDefaultFont", 10, "bold"))
                        item_label.pack(anchor="w")
                        if body_type == "object":
                            assert isinstance(item, dict), f"List item '{key}' must be a dict"
                            assert "props" in body[key], f"'props' not specified for key '{key}' in body"
                            self._build_tree(parent, item, body[key]["props"], item_key, indent + 2)
                        else:
                            # Primitive types in list
                            self._build_tree(
                                parent, {f"{key}_{i}": item}, {f"{key}_{i}": body[key]}, item_key, indent + 1
                            )
                    body[key]["isArray"] = True  # Restore isArray
                elif body_type == "object":
                    assert isinstance(value, dict), f"Value for key '{key}' must be a dict"
                    assert "props" in body[key], f"'props' not specified for key '{key}' in body"
                    # Expandable section for dict
                    label = ttk.Label(frame, text=f"▼ {key}:", font=("TkDefaultFont", 10, "bold"))
                    label.pack(anchor="w")
                    self._build_tree(parent, value, body[key]["props"], full_key, indent + 1)
                elif body_type == "bool":
                    _create_boolean_entry(
                        frame,
                        key,
                        value,
                        full_key,
                        self._notify_change,
                        self.boolean_vars,
                    )
                elif body_type == "multiline_string":
                    _create_multiline_text_widget(
                        parent,
                        frame,
                        key,
                        value,
                        full_key,
                        indent,
                        self._on_text_modified,
                        self.text_entries,
                    )

                elif body_type in ("string", "float", "int"):
                    label = ttk.Label(frame, text=f"{key}:", width=25, anchor="e")
                    label.pack(side="left")
                    if body_type == "string":
                        _create_string_entry(
                            frame,
                            key,
                            value,
                            full_key,
                            self._notify_change,
                            self.string_entries,
                        )
                    elif body_type in ("int", "float"):
                        _create_numeric_entry(
                            frame,
                            key,
                            value,
                            self.register,
                            self._notify_change,
                            body,
                            body_type,
                            full_key,
                            self.int_entries,
                            self.float_entries,
                        )
                    else:
                        raise ValueError(f"Unsupported body type: {body_type}")
                elif body_type == "file":
                    _create_file_entry(
                        frame,
                        key,
                        value,
                        full_key,
                        body,
                        self._notify_change,
                        self.file_entries,
                    )
                elif body_type == "combo":
                    _create_combo_entry(
                        frame,
                        key,
                        value,
                        full_key,
                        body,
                        self._notify_change,
                        self.combo_entries,
                    )
                else:
                    raise ValueError(f"Unsupported body type: {body_type}")
        except Exception as e:
            messagebox.showerror("Error", f"Error building JSON tree at prefix '{prefix}':\n{e}")
            logging.exception("Error building JSON tree at prefix '%s': %s", prefix, e)
            raise e

    def get_data(self) -> dict[str, Any]:
        """Get the current data from the editor."""
        result = self._deep_copy_structure(self.data)

        # Update string entries
        for full_key, entry in self.string_entries.items():
            self._set_nested_value(result, full_key, self._parse_value(entry.get()))

        # Update int entries
        for full_key, entry in self.int_entries.items():
            self._set_nested_value(result, full_key, self._parse_value(entry.get()))

        # Update float entries
        for full_key, entry in self.float_entries.items():
            self._set_nested_value(result, full_key, self._parse_value(entry.get()))

        # Update multiline text entries
        for full_key, text_widget in self.text_entries.items():
            text_value = text_widget.get("1.0", "end-1c")  # Get text without trailing newline
            self._set_nested_value(result, full_key, text_value)

        # Update boolean entries
        for full_key, var in self.boolean_vars.items():
            self._set_nested_value(result, full_key, var.get())

        # Update file entries
        for full_key, combo in self.file_entries.items():
            self._set_nested_value(result, full_key, combo.get())

        # Update combo entries
        for full_key, combo in self.combo_entries.items():
            self._set_nested_value(result, full_key, combo.get())

        return result

    def _deep_copy_structure(self, obj: Any) -> Any:
        """Deep copy maintaining structure."""
        if isinstance(obj, dict):
            return {k: self._deep_copy_structure(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy_structure(item) for item in obj]
        else:
            return obj

    def _set_nested_value(self, data: dict, key_path: str, value: Any) -> None:
        """Set a value in a nested dict using dot notation."""
        try:
            keys = key_path.split(".")
            current = data
            for key in keys[:-1]:
                # Handle list indices
                match = re.match(r"(\w+)\[(\d+)\]", key)
                if match:
                    list_key = match.group(1)
                    index = int(match.group(2))
                    assert list_key in current and isinstance(
                        current[list_key], list
                    ), f"List key '{list_key}' not found in data"
                    current = current[list_key][index]
                if key in current:
                    current = current[key]
            final_key = keys[-1]
            if final_key in current:
                current[final_key] = value
        except Exception as e:
            messagebox.showerror("Error", f"Error setting nested value for key '{key_path}':\n{e}")
            logging.exception("Error setting nested value for key '%s': %s", key_path, e)
            raise e

    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value to its appropriate type.

        Note: Boolean conversion is NOT done here because booleans are handled
        separately via BooleanVar checkboxes. Strings like "True" or "False"
        should remain as strings.
        """
        if value_str.lower() == "null" or value_str.lower() == "none":
            return None
        try:
            if "." in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            return value_str

    def _notify_change(self) -> None:
        """Notify that a value has changed."""
        if self._on_change:
            self._on_change(True)

    def _on_text_modified(self, event: tk.Event) -> None:
        """Handle text widget modification."""
        widget = event.widget
        if widget.edit_modified():
            self._notify_change()
            widget.edit_modified(False)


class ImageViewer(ttk.Frame):
    """Frame for displaying images."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.images: list[ImageTk.PhotoImage] = []  # Keep references

        # Create canvas with scrollbar
        self.canvas = tk.Canvas(self, highlightthickness=0, bg="#2b2b2b")
        self.scrollbar_y = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)

        self.scrollbar_y.pack(side="right", fill="y")
        self.scrollbar_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

    def display_images(self, image_paths: list[str]) -> None:
        """Display images from file paths."""
        self.images.clear()
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for _i, path in enumerate(image_paths):
            try:
                img = Image.open(path)
                # Resize to fit
                img.thumbnail((400, 400), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.images.append(photo)

                frame = ttk.Frame(self.scrollable_frame)
                frame.pack(side="left", padx=10, pady=10)

                label = ttk.Label(frame, image=photo)
                label.pack()

                name_label = ttk.Label(frame, text=os.path.basename(path), wraplength=400)
                name_label.pack()

            except Exception as e:
                error_label = ttk.Label(self.scrollable_frame, text=f"Error loading {path}: {e}")
                error_label.pack(side="left", padx=10, pady=10)
                logging.exception("Error loading image %s", path)

    def clear(self) -> None:
        """Clear all images."""
        self.images.clear()
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()


class JSONManagerApp:
    """Main application class."""

    @property
    def flow(self) -> Optional[Callable[[str, str, int], list[str]]]:
        """Get the flow callable."""
        return self._flow

    @flow.setter
    def flow(self, value: Optional[Callable[[str, str, int], list[str]]]) -> None:
        """Set the flow callable and validate its signature."""
        if value is None:
            self._flow = None
            return
        # Validate that value is a Callable
        assert callable(value), "flow must be a callable"
        # Validate the signature main(path_file: str, filename: str, steps: int) -> list[str]
        signature = inspect.signature(value)
        params = list(signature.parameters.keys())
        assert params == ["path_file", "filename", "steps"], f"Invalid parameters: {params}"
        assert signature.return_annotation == list[str], "Invalid return type"

        self._flow = value

    @flow.deleter
    def flow(self) -> None:
        """Delete the flow callable."""
        self._flow = None

    @property
    def flow_body(self) -> Optional[dict[str, Any]]:
        """Get the flow body from the current JSON data."""
        return self._flow_body

    @flow_body.setter
    def flow_body(self, filename: str) -> None:
        """Set the flow body."""
        if filename is None:
            self._flow_body = None
            return

        assert os.path.isfile(filename), f"{filename} is not a valid file"
        # Load YAML file
        with open(filename, "r", encoding="utf-8") as f:
            value: dict[str, Any] = yaml.safe_load(f)

        # Validate
        assert value is not None, "flow_body cannot be None"
        assert isinstance(value, dict), "flow_body must be a dictionary"
        assert "props" in value, "flow_body must contain 'props' key"

        self._flow_body = value

    @flow_body.deleter
    def flow_body(self) -> None:
        """Delete the flow body."""
        self._flow_body = None

    def __init__(self, root: tk.Tk):
        self._flow = None
        self._has_changes = False
        self.root = root
        self.root.title("JSON Configuration Manager")

        # Get screen dimensions and set window size
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self.current_file: str | None = None

        self._setup_ui()
        self._refresh_folder_list()

        # Intercept window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Top controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill="x", pady=(0, 10))

        # Flow selector
        ttk.Label(controls_frame, text="Flow Folder:").pack(side="left", padx=(0, 5))
        self.folder_var = tk.StringVar()
        self.folder_combo = ttk.Combobox(controls_frame, textvariable=self.folder_var, width=50, state="readonly")
        self.folder_combo.pack(side="left", padx=(0, 10))
        self.folder_combo.bind("<<ComboboxSelected>>", self._on_folder_selected)

        ttk.Button(controls_frame, text="Refresh Flows", command=self._refresh_folder_list).pack(side="left", padx=5)

        # File selector
        ttk.Label(controls_frame, text="JSON File:").pack(side="left", padx=(0, 5))
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(controls_frame, textvariable=self.file_var, width=50, state="readonly")
        self.file_combo.pack(side="left", padx=(0, 10))
        self.file_combo.bind("<<ComboboxSelected>>", self._on_file_selected)

        ttk.Button(controls_frame, text="Refresh JSONs", command=self._refresh_file_list).pack(side="left", padx=5)

        # Steps input
        ttk.Label(controls_frame, text="Steps:").pack(side="left", padx=(20, 5))
        self.steps_var = tk.StringVar(value="10")
        steps_entry = ttk.Entry(controls_frame, textvariable=self.steps_var, width=10)
        steps_entry.pack(side="left", padx=(0, 10))

        # Buttons
        ttk.Button(controls_frame, text="Save", command=self._save_file).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Save As", command=self._save_as_file).pack(side="left", padx=5)
        ttk.Button(controls_frame, text="Execute", command=self._execute).pack(side="left", padx=5)

        # Paned window for editor and images
        paned = ttk.PanedWindow(main_frame, orient="horizontal")
        paned.pack(fill="both", expand=True)

        # Left side - JSON Editor
        editor_frame = ttk.LabelFrame(paned, text="JSON Editor", padding="5")
        self.json_editor = JSONTreeEditor(editor_frame, on_change=self._mark_changes)
        self.json_editor.pack(fill="both", expand=True)
        paned.add(editor_frame, weight=1)

        # Right side - Image Viewer
        viewer_frame = ttk.LabelFrame(paned, text="Output Images", padding="5")
        self.image_viewer = ImageViewer(viewer_frame)
        self.image_viewer.pack(fill="both", expand=True)
        paned.add(viewer_frame, weight=1)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(fill="x", pady=(10, 0))

    def _mark_changes(self, toggle_changes: bool) -> None:
        """Mark that there are unsaved changes."""
        if toggle_changes:
            if not self._has_changes:
                self._has_changes = True
                self._update_title()
        else:
            self._has_changes = False
            self._update_title()

    def _update_title(self) -> None:
        """Update window title to reflect unsaved changes."""
        base_title = "JSON Configuration Manager"
        if self._has_changes:
            self.root.title(f"*{base_title} - Unsaved Changes")
        else:
            self.root.title(base_title)

    def _check_unsaved_changes(self) -> bool:
        """Check for unsaved changes and prompt user.

        Returns:
            True if it's safe to proceed (no changes or user chose to discard/save).
            False if user cancelled the operation.
        """
        if not self._has_changes:
            return True

        response = messagebox.askyesnocancel(
            "Unsaved Changes",
            (
                "You have unsaved changes.\n\nDo you want to save before continuing?\n\n"
                "If you choose 'No', changes will be discarded."
            ),
            icon="warning",
        )

        if response is None:  # Cancel
            return False
        elif response:  # Yes - save
            self._save_file()
            return not self._has_changes  # Return True only if save succeeded
        else:  # No - discard
            self._on_file_selected(skip_check=True)
            self._mark_changes(False)
            return True

    def _on_close(self) -> None:
        """Handle window close event."""
        if self._check_unsaved_changes():
            self.root.destroy()

    def _refresh_folder_list(self) -> None:
        """Refresh the list of Flow folders."""
        try:
            folders = [
                f
                for f in os.listdir(gui_utils.get_main_images_path())
                if os.path.isdir(os.path.join(gui_utils.get_main_images_path(), f))
            ]
            folders.sort()
            self.folder_combo["values"] = folders
            self.status_var.set(f"Found {len(folders)} Flow folders")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read directory: {e}")
            logging.exception("Failed to read Flow folders")

    def _refresh_file_list(self) -> None:
        """Refresh the list of JSON files."""
        foldername = self.folder_var.get()
        assert foldername, "Folder name is empty"
        try:
            look_path = os.path.join(gui_utils.get_main_images_path(), foldername)
            files = [f for f in os.listdir(look_path) if f.endswith(".json")]
            files.sort()
            self.file_combo["values"] = files
            self.status_var.set(f"Found {len(files)} JSON files")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read directory: {e}")
            logging.exception("Failed to read JSON files in folder %s", foldername)

    def _on_folder_selected(self, _event: tk.Event | None = None) -> None:
        """Handle file selection."""
        foldername = self.folder_var.get()
        if not foldername:
            return

        if not self._check_unsaved_changes():
            return

        filepath = os.path.join(gui_utils.get_main_images_path(), foldername)
        try:
            # validate that foldername is a directory
            assert os.path.isdir(filepath), f"{foldername} is not a valid directory"
            del self.flow
            script_path = gui_utils.get_main_script_path(foldername)

            def load_script(loading_win: tk.Toplevel = None) -> None:
                """Load the script for the selected folder and set the flow function."""
                try:
                    # verify that the script has a main function
                    module_path = Path(script_path)
                    spec = importlib.util.spec_from_file_location(module_path.stem, script_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    assert hasattr(module, "main"), f"Script {script_path} does not have a main function"
                    self.flow = getattr(module, "main")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load script:\n{e}")
                    logging.exception("Failed to load script for folder %s", foldername)
                    raise e
                finally:
                    if loading_win:
                        loading_win.destroy()

            loading_win = _show_loading_modal(self.root, message=f"Loading Flow: {foldername}...")
            t = threading.Thread(target=load_script, args=(loading_win,))
            t.start()
            while t.is_alive():
                self.root.after(100, self.root.update())
            assert self.flow is not None, "Flow function is not set after loading script"
            self.folder_var.set(foldername)

            # Set flow body
            flow_yaml_path = script_path.replace(".py", ".yml")
            self.flow_body = flow_yaml_path

            # Clear previous data
            self.json_editor.load_data({}, {"props": {}})
            self.current_file = None
            self._mark_changes(False)
            self.status_var.set(f"Selected folder: {foldername}")
            self.image_viewer.clear()
            self._refresh_file_list()
        except ModuleNotFoundError:
            messagebox.showerror("Error", f"Script not found for folder: {foldername}")
            logging.exception("Script not found for folder %s", foldername)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {foldername},\n{e}")
            logging.exception("Failed to load folder %s", foldername)

    def _on_file_selected(self, _event: tk.Event | None = None, skip_check: bool = False) -> None:
        """Handle file selection."""
        foldername = self.folder_var.get()
        assert foldername, "Folder name is empty"
        flow_body = self.flow_body
        assert flow_body is not None, "Flow body is not set"
        filename = self.file_var.get()
        if not filename:
            return

        if not skip_check and not self._check_unsaved_changes():
            return

        body = self.flow_body
        assert body is not None, "Flow body is not set"

        filepath = os.path.join(gui_utils.get_main_images_path(), foldername, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.json_editor.load_data(data, body)
            self.current_file = filepath
            self._mark_changes(False)
            self.status_var.set(f"Loaded: {filename}")
            self.image_viewer.clear()
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Failed to parse JSON:\n{e}")
            logging.exception("Failed to parse JSON file %s", filepath)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")
            logging.exception("Failed to load file %s", filepath)

    def _save_file(self) -> None:
        """Save changes to the current file."""
        if not self.current_file:
            messagebox.showwarning("Warning", "No file selected")
            return

        try:
            data = self.json_editor.get_data()
            with open(self.current_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._mark_changes(False)
            self.status_var.set(f"Saved: {os.path.basename(self.current_file)}")
            messagebox.showinfo("Success", "File saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
            logging.exception("Failed to save file %s", self.current_file)

    def _save_as_file(self) -> None:
        """Save as a new file with user provided name or UUID."""
        try:
            data = self.json_editor.get_data()

            filename = simpledialog.askstring("Save As", "Enter filename (leave blank for UUID):", parent=self.root)
            if filename is None:
                return

            filename = filename.strip()

            if not filename:
                new_uuid = str(uuid.uuid4())
                new_filename = f"{new_uuid}.json"
            else:
                if " " in filename:
                    messagebox.showerror("Error", "Spaces are not allowed in filename")
                    return
                new_filename = filename if filename.endswith(".json") else f"{filename}.json"
            foldername = self.folder_var.get()
            assert foldername, "Folder name is empty"
            new_filepath = os.path.join(gui_utils.get_main_images_path(), foldername, new_filename)

            with open(new_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._refresh_file_list()
            self.file_var.set(new_filename)
            self.current_file = new_filepath
            self._mark_changes(False)
            self.status_var.set(f"Saved as: {new_filename}")
            messagebox.showinfo("Success", f"File saved as:\n{new_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
            logging.exception("Failed to save file as new file")

    def _execute(self) -> None:
        """Execute the main function with the selected JSON."""
        if not self.current_file:
            messagebox.showwarning("Warning", "No file selected")
            return

        if not self._check_unsaved_changes():
            return

        try:
            steps = int(self.steps_var.get())
        except ValueError:
            messagebox.showerror("Error", "Steps must be a number")
            return

        # Get filename without extension
        filename = os.path.basename(self.current_file)
        foldername = self.folder_var.get()
        try:
            assert foldername, "Folder name is empty"
            assert foldername in self.current_file, "Folder name must be part of the current file path"
        except AssertionError as e:
            messagebox.showerror("Error", f"Failed to execute:\n{e}")
            logging.exception("Execution failed")
            raise e

        filename_without_ext = os.path.splitext(filename)[0]

        self.status_var.set(f"Executing with {filename_without_ext}...")
        self.root.update()

        def run_flow(loading_win: tk.Toplevel) -> None:
            """Run the flow function in a separate thread."""
            try:
                assert self.flow is not None, "Flow function is not set"
                # Import and run the main function
                assert self.flow is not None, "Flow function is not set"
                assert callable(self.flow), "Flow is not callable"
                flow_fn = cast(Callable[[str, str, int], list[str]], self.flow)

                with torch.inference_mode():
                    image_paths = flow_fn(
                        os.path.join(gui_utils.get_main_images_path(), foldername),
                        filename_without_ext,
                        steps,
                    )

                if image_paths:
                    self.image_viewer.display_images(image_paths)
                    self.status_var.set(f"Execution complete. Generated {len(image_paths)} images.")
                else:
                    self.status_var.set("Execution complete. No images generated.")
                    messagebox.showinfo("Info", "Execution completed but no images were generated.")

            except Exception as e:
                self.status_var.set("Execution failed")
                messagebox.showerror("Execution Error", f"Failed to execute:\n{e}")
                # log exception stack trace
                logging.exception("Execution failed")
                raise e
            finally:
                if loading_win:
                    loading_win.destroy()

        loading_win = _show_loading_modal(self.root, message=f"Executing Flow {foldername}: {filename_without_ext}...")
        threading.Thread(target=run_flow, args=(loading_win,)).start()


def main() -> None:
    """Main entry point."""
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    JSONManagerApp(root)
    root.mainloop()
