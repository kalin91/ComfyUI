
rm -rf /data/home2/kalin/repos/ComfyUI/tools/logs/*
/data/home2/kalin/repos/ComfyUI/.venv/bin/python3.12 /data/home2/kalin/repos/ComfyUI/main.py "$@" > /data/home2/kalin/repos/ComfyUI/tools/logs/execute_$(date +%Y%m%d_%H%M%S).log 2>&1
