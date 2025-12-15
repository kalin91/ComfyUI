""" "JSON Tree Editor GUI Component."""

import re
import os
import logging
import tkinter as tk
import random
from tkinter import ttk, messagebox
from typing import Any, Callable, Optional
from PIL import Image, ImageTk
import json_gui.utils as gui_utils
from json_gui.scroll_utils import bind_frame_scroll_events, bind_scroll_events
from json_gui.constants import COMBO_CONSTANTS, JSON_CANVAS_NAME, JSON_SCROLL_FRAME_NAME


def open_preview(file_path: str, frame: ttk.Widget) -> None:
    """Open a floating preview window for the selected image."""
    try:
        try:
            img = Image.open(file_path)
        except Exception as e:
            messagebox.showerror("Preview Error", f"Cannot open image:\n{e}")
            return

        parent_win = frame.winfo_toplevel()
        win = tk.Toplevel(parent_win)
        win.title(f"Preview - {os.path.basename(file_path)}")
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
    except Exception as e:
        logging.exception("Error opening preview window: %s", e)
        messagebox.showerror("Preview Error", f"Error opening preview:\n{e}")


def _create_string_entry(
    frame: ttk.Widget,
    key: str,
    value: Any,
    full_key: str,
    notify_change: Callable[[], None],
    string_entries: dict[str, tk.Entry],
) -> None:
    """Create a string entry widget."""
    try:
        entry = ttk.Entry(frame, width=60)
        entry.insert(0, str(value))
        entry.bind("<KeyRelease>", lambda e: notify_change())
        entry.pack(side="left", padx=5, fill="x", expand=True)
        assert isinstance(value, str), f"Value for key '{key}' must be a string"
        string_entries[full_key] = entry
    except Exception as e:
        logging.exception("Error creating string entry for key '%s': %s", key, e)


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
    try:
        label = ttk.Label(frame, text=f"{key}:", width=25, anchor="e")
        label.pack(side="left")
        var = tk.BooleanVar(value=value)
        check = ttk.Checkbutton(frame, variable=var, command=notify_change)
        check.pack(side="left", padx=5)
        boolean_vars[full_key] = var
    except Exception as e:
        logging.exception("Error creating boolean entry for key '%s': %s", key, e)
        raise e


def _create_open_preview_handler(p_combo: ttk.Combobox, p_folder: str, p_frame: ttk.Widget) -> Callable[[], None]:
    """Create a handler to open a preview window."""

    def _open_preview(folder=p_folder, combo=p_combo, frame=p_frame) -> None:
        """Open a floating preview window for the selected image."""
        path = os.path.join(folder, combo.get())
        if not path:
            messagebox.showwarning("Preview", "Select a file to preview")
            return
        open_preview(path, frame)

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
    try:
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

        # Bind mousewheel directly to text widget (not bind_all)
        bind_frame_scroll_events(combo, combo)

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
    except Exception as e:
        logging.exception("Error creating file entry for key '%s': %s", key, e)
        raise e


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
    try:
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

        # Bind mousewheel directly to text widget (not bind_all)
        bind_frame_scroll_events(combo, combo)

        combo.pack(side="left", padx=5, fill="x", expand=True)
        combo_entries[full_key] = combo
    except Exception as e:
        logging.exception("Error creating combo entry for key '%s': %s", key, e)
        raise e


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
    try:
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

        # Bind mousewheel directly to text widget (not bind_all)
        bind_scroll_events(text_widget)

        text_widget.pack(side="left", fill="both", expand=True)
        text_scrollbar.pack(side="right", fill="y")

        text_entries[full_key] = text_widget
    except Exception as e:
        logging.exception("Error creating multiline text widget for key '%s': %s", key, e)
        raise e


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
        try:
            max_decimals = 0 if b_type == "int" else int(p_format.replace("f", "").split(".")[-1])
            messagebox.showwarning(
                "Invalid Input",
                (
                    f"Please enter a valid {b_type} between {p_min} and "
                    f"{p_max} with up to {max_decimals} decimal places."
                ),
            )
            on_change()
        except Exception as e:
            logging.exception("Error handling invalid input: %s", e)
            on_change()
            raise e

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
        try:
            on_change()
            if bt == "int":
                val = random.randint(int(mn), int(mx))
                e.set(val)
            else:
                val = random.uniform(float(mn), float(mx))
                e.set(fmt % val)
        except Exception as ex:
            logging.exception("Error setting random value")
            raise ex

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
    try:
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

        # Bind mousewheel directly to text widget (not bind_all)
        bind_frame_scroll_events(entry, entry)

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
    except Exception as e:
        logging.exception("Error creating numeric entry for key '%s': %s", key, e)
        raise e


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
        self.canvas = tk.Canvas(self, highlightthickness=0, name=JSON_CANVAS_NAME)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, name=JSON_SCROLL_FRAME_NAME)

        self.scrollable_frame.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Add horizontal scrollbar
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")

        # Bind mousewheel only when mouse is over this widget
        bind_frame_scroll_events(self, self.canvas, True)

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
