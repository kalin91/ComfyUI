# ComfyUI JSON GUI

## Overview

**ComfyUI JSON GUI** is a specialized, lightweight application ("fork") built on top of ComfyUI. It serves as a **headless orchestrator** that allows users to execute complex ComfyUI workflows through a simplified, form-based graphical interface (Tkinter), rather than the traditional node-graph editor.

The primary goal of this project is to separate the *definition* of a workflow from its *execution*. By abstracting parameters into a structured JSON/YAML configuration, users can generate images and run tasks by simply modifying fields (prompts, models, resolutions) without needing to understand the underlying node graph.

## Features

* **Parameter-Based Interface**: A dynamic Tkinter GUI that builds forms based on YAML schemas.
* **Headless Execution**: Bypasses the standard ComfyUI WebSocket server, running nodes programmatically via a "Mimic" engine.
* **Pre-defined Flows**: Support for modular "Flows" (e.g., Stable Diffusion, ControlNet) that package logic and configuration together.
* **Integrated Image Viewer**: A simple built-in viewer to preview results immediately.
* **JSON/YAML Configuration**: Easy-to-read configuration files for storing and reloading experiment settings.

---

## How to Run

This application is designed to be run as a Python module. Below is the recommended Visual Studio Code Launch Configuration.

### VS Code Configuration

Add the following to your `.vscode/launch.json` or workspace configuration:

```json
{
  "name": "Python Debugger: Stable Diffussion",
  "type": "debugpy",
  "env": {
    "PYTHONPATH": "${workspaceFolder:main}"
  },
  "python": "${command:python.interpreterPath}",
  // "justMyCode": false,
  "console": "integratedTerminal",
  "request": "launch",
  "module": "main_json_gui",
  "args": "--temp-directory dummypath/temp --verbose DEBUG",
  "subProcess": true
}
```

### Command Line Arguments

* `--temp-directory`: Specifies the directory where temporary files and logs are stored.
* `--verbose`: Sets the logging level (e.g., `DEBUG`, `INFO`).

---

## Technical Architecture

The application follows a modular architecture distinct from the standard ComfyUI. It can be broken down into three main layers: The GUI (View), The Flow System (Controller/Model), and The Execution Engine (Backend).

### 1. Core Application Structure (`json_gui/`)

* **`main_json_gui.py`**: The entry point of the application. It bootstraps the Tkinter environment and initializes the main application window.
* **`json_gui/json_manager/`**: Contains the GUI logic.
  * **`json_manager_gui.py`**: The main controller for the window. It handles the loading of flows, saving of configurations, and dispatching the "Execute" command.
  * **`json_tree_editor.py`**: A crucial component that parses the `body.yml` schema of a flow and dynamically renders the corresponding UI widgets (Text boxes, Checkboxes, Dropdowns, File pickers). It creates a "Tree" of parameters that handles data validation.
  * **`image_viewer.py`**: A lightweight component to display generated images within the app.

### 2. The Flow System (`json_gui/scripts/`)

A "Flow" is a self-contained module representing a specific workflow (e.g., `controlnet_openpose`). Each flow resides in its own directory (e.g., `json_gui/scripts/controlnet_openpose/`) and consists of:

* **`body.yml`**: The **Schema**. It defines what the user sees in the GUI. It declares keys, default values, labels, and input types (e.g., `string`, `int`, `file_path`).
* **`flow.py`**: The **Logic**. Contains a class (inheriting from `AbsFlow`) that implements the `execution` method. This script reads the values from the GUI (passed as a dict) and orchestrates the ComfyUI nodes.
* **`model.py`** (Optional): Data classes or Pydantic models to strictly type the input configuration.

### 3. The Execution Engine ("Mimic" System)

This is the technical heart of the project. Since there is no web client involved, we cannot use the standard API. The app uses a set of scripts to "mimic" the behavior of the API server and execute nodes directly.

* **`json_gui/scripts/node_executor.py`**: Acts as the bridge between this app's logic and ComfyUI's internal node execution map. It finds the correct node class, instantiates it, and executes its main function.
* **`json_gui/scripts/mimic_classes.py` & `mimic_ksamplers.py`**: Wrappers around standard ComfyUI nodes. They handle the differences in processing (e.g., passing arguments directly rather than via JSON graph).
  * **Goal**: To allow Python code to say `KSampler.sample(...)` directly, handling the GPU resource management and VAE decoding internally.

---

## Workflow Lifecycle

1. **Selection**: User launches the app and uses the dropdown to select a Workflow (e.g., *ControlNet OpenPose*).
2. **UI Generation**:
    * `json_manager_gui.py` reads `json_gui/scripts/<flow_name>/body.yml`.
    * `json_tree_editor.py` dynamically builds the form widgets.
3. **Configuration**: User enters prompts, selects models, etc.
4. **Save**: The configuration is saved to a JSON file (the "Experiment").
5. **Execution**:
    * The user clicks **Execute**.
    * The corresponding `Flow` class in `flow.py` is instantiated.
    * The `execution(inputs)` method is called with the form data.
6. **Processing**:
    * The `Flow` logic prepares the tensors and calls the Mimic nodes.
    * ComfyUI's backend processes the image on the GPU.
7. **Output**:
    * The final image is saved.
    * The UI updates to show the result in the `ImageViewer`.

---

## Developer Guide: Creating a New Flow

To add a new workflow to the system:

1. **Create Directory**: Create a new folder in `json_gui/scripts/` (e.g., `my_new_flow`).
2. **Define Schema (`body.yml`)**:

    ```yaml
    - key: "positive_prompt"
      label: "Positive Prompt"
      type: "text"
      default: "A beautiful landscape"
    - key: "seed"
      label: "Seed"
      type: "int"
      default: 1234
    ```

3. **Implement Logic (`flow.py`)**:

    ```python
    from json_gui.utils import AbsFlow
    # Import your mimic nodes
    from json_gui.scripts.mimic_classes import CheckpointLoaderSimple, EmptyLatentImage

    class Flow(AbsFlow):
        def execution(self, inputs):
            # 1. Load Model
            model, clip, vae = CheckpointLoaderSimple().execute(inputs['checkpoint_name'])
            
            # 2. Logic...
            # ...
            
            return [saved_image_path] 
    ```

4. **Restart**: Restart the application. Your new flow will appear in the selection list.
