"""GUI Application for managing JSON configuration files."""

import json
import os
from typing import Any
import uuid
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import torch
import json_manager_starter
import logging

# Path to JSON files
PATH_TO_JSONS = "./images"


class JSONTreeEditor(ttk.Frame):
    """A hierarchical, editable view for JSON data."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.data: dict[str, Any] = {}
        self.entries: dict[str, tk.Entry] = {}
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

    def load_data(self, data: dict[str, Any]) -> None:
        """Load JSON data into the editor."""
        self.data = data
        self.entries.clear()
        self.text_entries.clear()
        self.boolean_vars.clear()
        self.list_entries.clear()

        # Clear existing widgets
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self._build_tree(self.scrollable_frame, data, "")

    def _build_tree(self, parent: tk.Widget, data: dict[str, Any], prefix: str, indent: int = 0) -> None:
        """Recursively build the tree view."""
        # List of keys that should use multiline text boxes
        multiline_keys = {"positive", "negative"}

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            frame = ttk.Frame(parent)
            frame.pack(fill="x", padx=(indent * 20, 5), pady=2)

            if isinstance(value, dict):
                # Expandable section for dict
                label = ttk.Label(frame, text=f"▼ {key}:", font=("TkDefaultFont", 10, "bold"))
                label.pack(anchor="w")
                self._build_tree(parent, value, full_key, indent + 1)

            elif isinstance(value, list):
                # List of items
                label = ttk.Label(
                    frame, text=f"▼ {key}: (list with {len(value)} items)", font=("TkDefaultFont", 10, "bold")
                )
                label.pack(anchor="w")
                self.list_entries[full_key] = []

                for i, item in enumerate(value):
                    item_key = f"{full_key}[{i}]"
                    if isinstance(item, dict):
                        item_frame = ttk.LabelFrame(parent, text=f"{key}[{i}]")
                        item_frame.pack(fill="x", padx=((indent + 1) * 20, 5), pady=5)
                        item_entries: dict[str, Any] = {}
                        for sub_key, sub_value in item.items():
                            sub_frame = ttk.Frame(item_frame)
                            sub_frame.pack(fill="x", padx=5, pady=2)
                            sub_label = ttk.Label(sub_frame, text=f"{sub_key}:", width=25, anchor="e")
                            sub_label.pack(side="left")

                            if isinstance(sub_value, bool):
                                var = tk.BooleanVar(value=sub_value)
                                check = ttk.Checkbutton(sub_frame, variable=var)
                                check.pack(side="left", padx=5)
                                item_entries[sub_key] = var
                            else:
                                entry = ttk.Entry(sub_frame, width=40)
                                entry.insert(0, str(sub_value))
                                entry.pack(side="left", padx=5, fill="x", expand=True)
                                item_entries[sub_key] = entry
                        self.list_entries[full_key].append(item_entries)
                    else:
                        item_frame = ttk.Frame(parent)
                        item_frame.pack(fill="x", padx=((indent + 1) * 20, 5), pady=2)
                        label = ttk.Label(item_frame, text=f"[{i}]:", width=25, anchor="e")
                        label.pack(side="left")

                        if isinstance(item, bool):
                            var = tk.BooleanVar(value=item)
                            check = ttk.Checkbutton(item_frame, variable=var)
                            check.pack(side="left", padx=5)
                            self.list_entries[full_key].append({"_value": var})
                        else:
                            entry = ttk.Entry(item_frame, width=40)
                            entry.insert(0, str(item))
                            entry.pack(side="left", padx=5, fill="x", expand=True)
                            self.list_entries[full_key].append({"_value": entry})

            elif key in multiline_keys:
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

            else:
                # Simple key-value
                label = ttk.Label(frame, text=f"{key}:", width=25, anchor="e")
                label.pack(side="left")

                if isinstance(value, bool):
                    var = tk.BooleanVar(value=value)
                    check = ttk.Checkbutton(frame, variable=var)
                    check.pack(side="left", padx=5)
                    self.boolean_vars[full_key] = var
                else:
                    entry = ttk.Entry(frame, width=60)
                    entry.insert(0, str(value))
                    entry.pack(side="left", padx=5, fill="x", expand=True)
                    self.entries[full_key] = entry

    def get_data(self) -> dict[str, Any]:
        """Get the current data from the editor."""
        result = self._deep_copy_structure(self.data)

        # Update simple entries
        for full_key, entry in self.entries.items():
            self._set_nested_value(result, full_key, self._parse_value(entry.get()))

        # Update multiline text entries
        for full_key, text_widget in self.text_entries.items():
            text_value = text_widget.get("1.0", "end-1c")  # Get text without trailing newline
            self._set_nested_value(result, full_key, text_value)

        # Update boolean entries
        for full_key, var in self.boolean_vars.items():
            self._set_nested_value(result, full_key, var.get())

        # Update list entries
        for full_key, items in self.list_entries.items():
            new_list = []
            for item_entries in items:
                if "_value" in item_entries:
                    widget = item_entries["_value"]
                    if isinstance(widget, tk.BooleanVar):
                        new_list.append(widget.get())
                    else:
                        new_list.append(self._parse_value(widget.get()))
                else:
                    new_item = {}
                    for sub_key, widget in item_entries.items():
                        if isinstance(widget, tk.BooleanVar):
                            new_item[sub_key] = widget.get()
                        else:
                            new_item[sub_key] = self._parse_value(widget.get())
                    new_list.append(new_item)
            self._set_nested_value(result, full_key, new_list)

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

    def __init__(self, root: tk.Tk):
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
        self._refresh_file_list()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Top controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill="x", pady=(0, 10))

        # File selector
        ttk.Label(controls_frame, text="JSON File:").pack(side="left", padx=(0, 5))
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(controls_frame, textvariable=self.file_var, width=50, state="readonly")
        self.file_combo.pack(side="left", padx=(0, 10))
        self.file_combo.bind("<<ComboboxSelected>>", self._on_file_selected)

        ttk.Button(controls_frame, text="Refresh", command=self._refresh_file_list).pack(side="left", padx=5)

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

    def _refresh_file_list(self) -> None:
        """Refresh the list of JSON files."""
        try:
            if not os.path.exists(PATH_TO_JSONS):
                os.makedirs(PATH_TO_JSONS)

            files = [f for f in os.listdir(PATH_TO_JSONS) if f.endswith(".json")]
            files.sort()
            self.file_combo["values"] = files
            self.status_var.set(f"Found {len(files)} JSON files")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read directory: {e}")

    def _on_file_selected(self, event: tk.Event | None = None) -> None:
        """Handle file selection."""
        filename = self.file_var.get()
        if not filename:
            return

        filepath = os.path.join(PATH_TO_JSONS, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.json_editor.load_data(data)
            self.current_file = filepath
            self.status_var.set(f"Loaded: {filename}")
            self.image_viewer.clear()
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Failed to parse JSON:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

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

            new_filepath = os.path.join(PATH_TO_JSONS, new_filename)

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
        filename_without_ext = os.path.splitext(filename)[0]

        self.status_var.set(f"Executing with {filename_without_ext}...")
        self.root.update()

        try:
            # Import and run the main function
            from script_controlnet import main as run_controlnet

            with torch.inference_mode():
                image_paths = run_controlnet(filename_without_ext, steps)

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
            raise e


def main() -> None:
    """Main entry point."""
    root = tk.Tk()

    json_manager_starter.apply_custom_paths()

    # Set theme
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")

    app = JSONManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
