# PeopleCounter

Line-crossing people counter built on Ultralytics YOLO26 + ByteTrack. Detects and tracks people across video frames and tallies in/out counts as they cross a defined line, writing an annotated output video.

## Pipeline

1. **Detection + tracking** — Ultralytics `ObjectCounter` runs YOLO26 detection restricted to the `person` class, tracked frame-to-frame with ByteTrack.
2. **Line-crossing counting** — a virtual line (or polygon region) is defined in frame coordinates; the counter tallies in/out crossings per tracked identity.
3. **Output** — an annotated video with bounding boxes, track IDs, and running counts, plus a final in/out/classwise count summary.

## Structure

```
peoplecounter/
  counter.py   PeopleCounter class: wraps ObjectCounter, handles video I/O safely
demo/
  run.py       CLI entry point
```

## Usage

```bash
pip install -r requirements.txt

python demo/run.py --input input.mp4 --output people_counted.mp4
```

By default the counting line is placed automatically at 60% of the frame height, spanning the middle 84% of the frame width, scaled to the actual input video's resolution. To set it manually:

```bash
python demo/run.py --input input.mp4 --output people_counted.mp4 --line-start 100,400 --line-end 1180,400
```

Add `--display` to preview the annotated video in a window while processing (requires a display; omit on headless machines/servers).

## Notes on this version

- The counting line auto-scales to the input video's resolution instead of using fixed pixel coordinates, which otherwise land in the wrong place (or off-frame) on videos of a different size.
- Video capture/writer are released in a `finally` block, so a mid-loop error doesn't leak file handles.
- `cv2.imshow`/`waitKey` are now optional (`--display`), since they crash on headless environments with no display.
- Final `in_count` / `out_count` / `classwise_count` are returned and printed instead of being discarded.
- CLI args replace hardcoded paths and model name.

## License

MIT
