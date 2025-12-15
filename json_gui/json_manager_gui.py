"""GUI Application for managing JSON configuration files."""

import importlib
import inspect
import json
import os
import logging
from pathlib import Path
from typing import Any, Callable, Optional, cast
import uuid
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import yaml
from PIL import Image, ImageTk
import torch
from app.logger import setup_logger
from json_gui.json_tree_editor import JSONTreeEditor
from json_gui.scroll_utils import bind_frame_scroll_events
from json_gui import loading_modal, utils as gui_utils
from comfy.cli_args import args


logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)


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

        # Bind mousewheel only when mouse is over this widget
        bind_frame_scroll_events(self, self.canvas, True)

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
            logging.debug("Refreshing JSON file list in folder: %s", look_path)
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

            def load_script() -> None:
                """Load the script for the selected folder and set the flow function."""
                # verify that the script has a main function
                module_path = Path(script_path)
                spec = importlib.util.spec_from_file_location(module_path.stem, script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                assert hasattr(module, "main"), f"Script {script_path} does not have a main function"
                self.flow = getattr(module, "main")


            loading_modal.show_loading_modal(self.root, load_script, (), f"Loading Flow: {foldername}...")
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

        def run_flow() -> None:
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
                raise e

        loading_modal.show_loading_modal(
            self.root, run_flow, (), f"Executing Flow {foldername}: {filename_without_ext}...", True
        )


def main() -> None:
    """Main entry point."""
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    JSONManagerApp(root)
    root.mainloop()
