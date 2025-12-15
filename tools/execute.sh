
rm -rf /data/home2/kalin/repos/ComfyUI/tools/logs/*
/data/home2/kalin/repos/ComfyUI/.venv/bin/python3.12 /data/home2/kalin/repos/ComfyUI/main.py "$@" 2>&1 | tee /data/home2/kalin/repos/ComfyUI/tools/logs/execute_$(date +%Y%m%d_%H%M%S).log
