# Fire-VAD Segmentation Tool

A desktop tool for dividing the videos of the fire-VAD dataset (RGB and thermal/IR
camera) into segments. It is the annotation tool without the caption fields: the
annotator only marks the frame where each new segment starts.

## Start

Needs Python 3.9+ with **Tk 8.6+**. The launchers pick a suitable Python, install
`numpy` and `pillow` into a private `.venv-segmenter` folder (first run only) and start the tool:

| System | Launcher |
|---|---|
| macOS | double-click `run_mac.command` |
| Linux | `bash run_linux.sh` |
| Windows | double-click `run_windows.bat` |

Or by hand:

```bash
pip install -r requirements.txt
python segment_gui.py
```

The tool looks for the folder `test_VAD` next to itself (or one or two folders up).
If it is not found, it asks you to select it.

## First start

You are asked for your **full name**. It identifies your files, so always type it the same way.
**Change annotator…** changes it later; **Change dataset folder…** changes the folder with the videos.

Your work is saved next to the tool (in `~/FireVAD-Segmenter` in a packaged build):

```
annotator_config.json                              saved name and videos location
annotations/<your_name>/environment_<X>/<Y>.json   one file per video
```

Each file holds `video_id`, `environment`, `annotator_id` and the list of segments with
`segment_id`, `start_frame`, `end_frame`, `temp_min_c` and `temp_max_c`
(the IR temperature range of the segment, filled in automatically).
The videos are never modified.

## Segmenting

1. Select a scene and press **Open scene**. Three synchronized panels open:
   **RGB**, **IR – relative** (contrast stretched per frame) and
   **IR – absolute** (fixed temperature scale: type the range or press **Video range**).
   Hovering over an IR panel shows the exact temperature in °C.
2. Pause at the frame where a new segment starts and press **B**. A new segment
   starts when something appears or disappears, or when a person changes their action.
   Segments are consecutive and cover the whole video. A video in which nothing
   changes is one segment.
3. Select a segment in the list to jump to its start. **Move start ◀ -1 / +1 ▶**
   shifts its start by one frame; **Remove start boundary** merges it into the previous segment.
4. Save with **Ctrl+S** (macOS: **Cmd+S**). You can stop at any time and continue later.
5. When you are done, click **Export all my annotations**. It writes
   `combined_export.json` into your annotation folder.

| Key | Action |
|---|---|
| Space | play / pause |
| ← / → | one frame back / forward |
| ↑ / ↓ | 10 frames back / forward |
| Home / End | first / last frame |
| B | a new segment starts at the current frame |
| Delete | remove the start boundary of the selected segment |
| F1 | help |

## Checks (developers)

```bash
python segment_gui.py --selftest --dataset-root /path/to/test_VAD   # headless data/logic checks
python segment_gui.py --guitest  --dataset-root /path/to/test_VAD   # scripted end-to-end GUI test
```
