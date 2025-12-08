"""GUI Application for managing JSON configuration files."""

import importlib
import inspect
import json
import os
import logging
import yaml
from pathlib import Path
import threading
from typing import Any, Callable, Optional, cast
import uuid
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import torch
from app.logger import setup_logger
import json_gui.json_manager_starter as json_manager_starter
from comfy.cli_args import args

logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)


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

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.data: dict[str, Any] = {}
        self.string_entries: dict[str, tk.Entry] = {}
        self.int_entries: dict[str, tk.Entry] = {}
        self.float_entries: dict[str, tk.Entry] = {}
        self.text_entries: dict[str, tk.Text] = {}  # For multiline text widgets
        self.boolean_vars: dict[str, tk.BooleanVar] = {}
        self.list_entries: dict[str, list[dict[str, Any]]] = {}

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

        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        assert "props" in body, "'props' key not found in body"

        self._build_tree(self.scrollable_frame, data, body["props"], "")

    def _build_tree(
        self, parent: tk.Widget, data: dict[str, Any], body: dict[str, Any], prefix: str, indent: int = 0
    ) -> None:
        """Recursively build the tree view."""
        # List of keys that should use multiline text boxes

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
                        self._build_tree(parent, {f"{key}_{i}": item}, {f"{key}_{i}": body[key]}, item_key, indent + 1)
                body[key]["isArray"] = True  # Restore isArray
            elif body_type == "object":
                assert isinstance(value, dict), f"Value for key '{key}' must be a dict"
                assert "props" in body[key], f"'props' not specified for key '{key}' in body"
                # Expandable section for dict
                label = ttk.Label(frame, text=f"▼ {key}:", font=("TkDefaultFont", 10, "bold"))
                label.pack(anchor="w")
                self._build_tree(parent, value, body[key]["props"], full_key, indent + 1)
            elif body_type == "bool":
                assert isinstance(value, bool), f"Value for key '{key}' must be a bool in list item '{key}'"
                label = ttk.Label(frame, text=f"{key}:", width=25, anchor="e")
                label.pack(side="left")
                var = tk.BooleanVar(value=value)
                check = ttk.Checkbutton(frame, variable=var)
                check.pack(side="left", padx=5)
                self.boolean_vars[full_key] = var
            elif body_type == "multiline_string":
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

                text_widget.pack(side="left", fill="both", expand=True)
                text_scrollbar.pack(side="right", fill="y")

                self.text_entries[full_key] = text_widget

            elif body_type == "string" or body_type == "float" or body_type == "int":
                label = ttk.Label(frame, text=f"{key}:", width=25, anchor="e")
                label.pack(side="left")
                entry = ttk.Entry(frame, width=60)
                entry.insert(0, str(value))
                entry.pack(side="left", padx=5, fill="x", expand=True)
                if body_type == "string":
                    assert isinstance(value, str), f"Value for key '{key}' must be a string"
                    self.string_entries[full_key] = entry
                elif body_type == "int":
                    assert isinstance(value, int), f"Value for key '{key}' must be an int"
                    self.int_entries[full_key] = entry
                elif body_type == "float":
                    assert isinstance(value, float), f"Value for key '{key}' must be a float"
                    self.float_entries[full_key] = entry
                else:
                    raise ValueError(f"Unsupported body type: {body_type}")

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
        keys = key_path.split(".")
        current = data
        for key in keys[:-1]:
            if key in current:
                current = current[key]
        final_key = keys[-1]
        if final_key in current:
            current[final_key] = value

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

        for i, path in enumerate(image_paths):
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
        self.json_editor = JSONTreeEditor(editor_frame)
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

    def _refresh_folder_list(self) -> None:
        """Refresh the list of Flow folders."""
        try:
            folders = [
                f
                for f in os.listdir(json_manager_starter.get_main_images_path())
                if os.path.isdir(os.path.join(json_manager_starter.get_main_images_path(), f))
            ]
            folders.sort()
            self.folder_combo["values"] = folders
            self.status_var.set(f"Found {len(folders)} Flow folders")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read directory: {e}")

    def _refresh_file_list(self) -> None:
        """Refresh the list of JSON files."""
        foldername = self.folder_var.get()
        assert foldername, "Folder name is empty"
        try:
            look_path = os.path.join(json_manager_starter.get_main_images_path(), foldername)
            files = [f for f in os.listdir(look_path) if f.endswith(".json")]
            files.sort()
            self.file_combo["values"] = files
            self.status_var.set(f"Found {len(files)} JSON files")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read directory: {e}")

    def _on_folder_selected(self, event: tk.Event | None = None) -> None:
        """Handle file selection."""
        foldername = self.folder_var.get()
        if not foldername:
            return

        filepath = os.path.join(json_manager_starter.get_main_images_path(), foldername)
        try:
            # validate that foldername is a directory
            assert os.path.isdir(filepath), f"{foldername} is not a valid directory"
            del self.flow
            script_path = json_manager_starter.get_main_script_path(foldername)

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
            self.status_var.set(f"Selected folder: {foldername}")
            self.image_viewer.clear()
            self._refresh_file_list()
        except ModuleNotFoundError:
            messagebox.showerror("Error", f"Script not found for folder: {foldername}")
            logging.exception("Script not found for folder %s", foldername)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {foldername},\n{e}")
            logging.exception("Failed to load folder %s", foldername)

    def _on_file_selected(self, event: tk.Event | None = None) -> None:
        """Handle file selection."""
        foldername = self.folder_var.get()
        assert foldername, "Folder name is empty"
        flow_body = self.flow_body
        assert flow_body is not None, "Flow body is not set"
        filename = self.file_var.get()
        if not filename:
            return

        body = self.flow_body
        assert body is not None, "Flow body is not set"

        filepath = os.path.join(json_manager_starter.get_main_images_path(), foldername, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.json_editor.load_data(data, body)
            self.current_file = filepath
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
            self.status_var.set(f"Saved: {os.path.basename(self.current_file)}")
            messagebox.showinfo("Success", "File saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

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
            new_filepath = os.path.join(json_manager_starter.get_main_images_path(), foldername, new_filename)

            with open(new_filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self._refresh_file_list()
            self.file_var.set(new_filename)
            self.current_file = new_filepath
            self.status_var.set(f"Saved as: {new_filename}")
            messagebox.showinfo("Success", f"File saved as:\n{new_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def _execute(self) -> None:
        """Execute the main function with the selected JSON."""
        if not self.current_file:
            messagebox.showwarning("Warning", "No file selected")
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
                        os.path.join(json_manager_starter.get_main_images_path(), foldername),
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
