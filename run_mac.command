#!/bin/bash
# Fire-VAD segmentation tool - launcher for macOS (and Linux via run_linux.sh).
#
# macOS:  double-click this file; the segmentation tool starts in Terminal.
# Terminal:  bash run_mac.command                  start the tool
#            bash run_mac.command --check-runtime  show the selected Python, change nothing
#            bash run_mac.command --selftest       headless check
#
# Needs Python 3.9+ with Tk 8.6+. Apple's /usr/bin/python3 ships Tk 8.5, which
# can show a blank window on modern macOS even though importing tkinter works.
# numpy and Pillow are installed into .venv-segmenter next to this script,
# never globally.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
VENV_DIR="$SCRIPT_DIR/.venv-segmenter"
OS_NAME="$(uname -s)"

finish() {
    STATUS=$?
    if [ "$STATUS" -ne 0 ]; then
        echo ""
        echo "The tool could not start or exited with an error (code $STATUS)."
        if [ -t 0 ]; then
            read -r -p "Press Enter to close..." _ || true
        fi
    fi
}
trap finish EXIT

echo "Fire-VAD segmentation tool"
echo "========================"

if [ ! -f "$SCRIPT_DIR/segment_gui.py" ]; then
    echo "ERROR: segment_gui.py not found next to this script."
    echo "Extract (unzip) the whole downloaded folder first and start the launcher from there."
    exit 1
fi

# ---- select a supported Python/Tk runtime --------------------------------
probe_python() {
    RUNTIME_REPORT=$("$1" -c '
import sys
if sys.version_info < (3, 9):
    sys.exit("Python 3.9 or newer is required.")
try:
    import tkinter
except ImportError:
    sys.exit("This Python has no tkinter support.")
if tkinter.TkVersion < 8.6:
    sys.exit("Tk %s is too old; Tk 8.6+ is required to avoid blank windows." % tkinter.TkVersion)
print("Python %s / Tk %s" % (sys.version.split()[0], tkinter.TkVersion))
' 2>&1)
}

python_install_help() {
    if [ "$OS_NAME" = "Darwin" ]; then
        echo "Install Python 3.13 from https://www.python.org/downloads/macos/"
        echo "(macOS 64-bit universal2 installer), then double-click this launcher again."
        echo "Homebrew users can instead run: brew install python-tk@3.13"
        echo "Apple's /usr/bin/python3 cannot be used: its Tk 8.5 shows blank windows."
        if [ -t 0 ]; then
            echo "Opening the Python download page..."
            open "https://www.python.org/downloads/macos/" >/dev/null 2>&1 || true
        fi
    else
        echo "Install Python 3 with Tk and venv support, then run this launcher again:"
        echo "  Ubuntu/Debian:  sudo apt install python3 python3-tk python3-venv"
        echo "  Fedora:         sudo dnf install python3 python3-tkinter"
        echo "  Arch:           sudo pacman -S python tk"
    fi
}

PY=""
if [ -n "${FIREVAD_PYTHON:-}" ]; then
    # An explicit override must fail clearly, not silently choose another Python.
    CAND=$(command -v "$FIREVAD_PYTHON") || {
        echo "ERROR: FIREVAD_PYTHON not found: $FIREVAD_PYTHON"
        exit 1
    }
    if ! probe_python "$CAND"; then
        echo "ERROR: $CAND: $RUNTIME_REPORT"
        exit 1
    fi
    PY="$CAND"
else
    # Explicit paths also work with Finder's minimal PATH. Prefer modern
    # framework/Homebrew installs to /usr/bin/python3.
    for CAND in "$VENV_DIR/bin/python" \
        /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
        /Library/Frameworks/Python.framework/Versions/3.*/bin/python3 \
        /opt/homebrew/bin/python3 /usr/local/bin/python3 \
        /opt/homebrew/opt/python@3.*/bin/python3.* \
        /usr/local/opt/python@3.*/bin/python3.* \
        python3 python /usr/bin/python3; do
        RESOLVED=$(command -v "$CAND" 2>/dev/null) || continue
        case "$RESOLVED" in *-config) continue ;; esac
        if probe_python "$RESOLVED"; then
            PY="$RESOLVED"
            break
        fi
        echo "Skipping $RESOLVED: $RUNTIME_REPORT"
    done
fi

if [ -z "$PY" ]; then
    echo "ERROR: no Python 3.9+ with Tk 8.6+ was found."
    python_install_help
    exit 1
fi

echo "Selected: $PY ($RUNTIME_REPORT)"
if [ "${1:-}" = "--check-runtime" ]; then
    # Diagnostic only: no GUI, config writes, venv creation, or pip installs.
    exit 0
fi

# ---- install packages into an isolated local environment -----------------
has_packages() {
    "$1" -c "import numpy; from PIL import Image, ImageTk" >/dev/null 2>&1
}

if ! has_packages "$PY"; then
    if [ "$PY" != "$VENV_DIR/bin/python" ] && [ -e "$VENV_DIR" ]; then
        # Reached with FIREVAD_PYTHON: reuse the environment created from it before.
        if [ -x "$VENV_DIR/bin/python" ] && probe_python "$VENV_DIR/bin/python"; then
            PY="$VENV_DIR/bin/python"
        else
            echo "ERROR: $VENV_DIR exists but is not a usable segmentation environment."
            echo "It has been left untouched. Delete the .venv-segmenter folder and start again."
            exit 1
        fi
    fi
    if [ "$PY" != "$VENV_DIR/bin/python" ]; then
        echo "Creating the segmentation environment in $VENV_DIR ..."
        if ! "$PY" -m venv "$VENV_DIR"; then
            rm -rf "$VENV_DIR"      # created by this run; leave nothing half-built
            echo "ERROR: could not create a Python virtual environment."
            if [ "$OS_NAME" != "Darwin" ]; then
                echo "Ubuntu/Debian: sudo apt install python3-venv"
            fi
            exit 1
        fi
        PY="$VENV_DIR/bin/python"
    fi
    if ! has_packages "$PY"; then
        echo "Installing numpy and Pillow for this tool (first run only, needs internet)..."
        if ! "$PY" -m pip install --disable-pip-version-check --only-binary=:all: \
                -r "$SCRIPT_DIR/requirements.txt"; then
            echo "ERROR: installing numpy and Pillow failed. Check the internet connection."
            echo "If this Python is very new, install Python 3.13 instead and delete .venv-segmenter."
            exit 1
        fi
        "$PY" -c "import numpy; from PIL import Image, ImageTk" || exit 1
    fi
fi

# ---- launch -----------------------------------------------------------------
echo ""
echo "Starting with: $PY"
echo "Keep this window open until you close the segmentation tool."
echo ""
"$PY" "$SCRIPT_DIR/segment_gui.py" "$@"
exit $?
