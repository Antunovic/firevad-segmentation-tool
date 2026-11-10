#!/usr/bin/env bash
# Fire-VAD segmentation tool - Linux launcher. run_mac.command holds the logic
# and works on Linux as well.
#   bash run_linux.sh                  start the tool
#   bash run_linux.sh --check-runtime  show the selected Python, change nothing
exec bash "$(cd "$(dirname "$0")" && pwd)/run_mac.command" "$@"
