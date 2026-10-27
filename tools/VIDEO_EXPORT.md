# Offline experiment video export

`export_replay_video.mjs` reads a stopped run's `manifest.json`, `state.json`, and
`events.jsonl`, then embeds a snapshot in a **separate** presentation copy of
`codetest/viewer.html`. It renders 1920 × 1080 still-frame scenes with headless
Playwright and joins them with local ffmpeg (H.264, 30 fps, no audio).

It does not import the Python engine, call a model, invoke `report.replay`, write
run reports or metrics, or modify the source run. Browser HTTP(S) requests are
blocked. Source file hashes are checked before and after rendering and included
in `export-manifest.json`, with template/exporter/video hashes and every scene.

The output folder must be new and outside the run. A `running` source is refused;
other incomplete statuses require `--allow-partial` and produce prominent partial
labels. A run that changes during rendering fails verification. Do not hand off
an output as successful without `export-manifest.json`.

## Usage

Required: `--run SOURCE_RUN --out NEW_OUTPUT_DIRECTORY`.

Optional: `--allow-partial`, `--title TEXT`, `--viewer VIEWER_HTML`,
`--playwright PLAYWRIGHT_MODULE_DIRECTORY`, `--browser CHROMIUM_EXECUTABLE`,
`--ffmpeg FFMPEG_EXECUTABLE`. The default viewer is from the source run's repo;
the default Playwright module and ffmpeg binary use normal Node/PATH resolution.

A portable example, after installing Playwright and ffmpeg locally:

```text
node tools/export_replay_video.mjs --run /path/to/completed-run --out /path/to/new-video --title "Experiment replay"
```

For an incomplete stopped run, add `--allow-partial`. Invoke separately for each
completed condition to obtain videos in the same style. No comparison result is
inferred or fabricated. Windows callers can pass absolute Windows paths or UNC
paths to WSL files and explicit dependency paths through the options above.

## Interpretation

These are **recorded round summaries with selected event excerpts**, not a
continuous reconstruction of state after every action. A round's snapshot and
recorded spend may summarize the entire recorded round while an earlier event
is highlighted. Partial rounds can use the previous snapshot plus recorded
membership. The score trajectory shows the entire recorded history. Unknown
scores are shown as missing, never changed to zero. Visible development tests
are not hidden generalization scores. Virtual wealth is distinct from API USD;
the main cost panel totals recorded `api_charged` usage estimates through the
selected round, **not confirmed bills**. Its hint separately displays the whole
run's `manifest.usage.accounted_usd`, including unresolved reserves, when present.
This accounting total is not specific to the selected round. The exporter does
not reconcile unconfirmed charges. Source metrics are read only for the reused viewer.

The shortest preview has initialization, formation, highest-bid candidate,
publication/stop highlights, and a final summary. Ties use original event order;
this choice has no effect on the experiment or the market algorithm.

## Validation

```text
node --check tools/export_replay_video.mjs
node --test tools/test_export_replay_video.mjs
```

Tests use temporary local fixtures and reject running runs, unmarked partial
exports, source-contained output, and discontinuous logs before rendering. An
actual interrupted-run export was also inspected visually and checked with
ffprobe for 1920 × 1080, H.264, 30 fps and manifest-matching duration.
