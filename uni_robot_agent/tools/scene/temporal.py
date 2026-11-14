"""Render real temporal RGB with explicit English provenance for every image."""
from datetime import datetime, timezone


def temporal_frames(manifest, cameras):
    history = manifest.get("history")
    if not history:
        return []
    names = list(dict.fromkeys(cameras))
    times = [tuple(row["simulation_time_s"] for row in history[name]) for name in names]
    if not times or any(row != times[0] for row in times):
        raise ValueError("Temporal RGB camera samples must be synchronized")
    rows = []
    for group in range(len(times[0])):
        for name in names:
            sample = dict(history[name][group], camera=name)
            if group == len(times[0]) - 1:
                if sample["state_version"] != manifest["state_version"]:
                    raise ValueError("Temporal RGB must end at the current state")
                # The current image/calibration may be a fresh high-resolution capture.
                sample.update(rgb=manifest["cameras"][name]["rgb"], captured_at=manifest["captured_at"],
                              frame_id=f"{manifest['snapshot_id']}:{name}:current")
            sample["image_index"] = len(rows)
            sample["captured_at_utc"] = datetime.fromtimestamp(sample["captured_at"], timezone.utc).isoformat()
            rows.append(sample)
    return rows


def frame_label(frame):
    return (f"[RUA RGB frame {frame['image_index']:03d}] camera={frame['camera']}; "
            f"frame_id={frame['frame_id']}; simulation_time_s={frame['simulation_time_s']:.3f}; "
            f"captured_at_utc={frame['captured_at_utc']}; state_version={frame['state_version']}. "
            "Frames are ordered oldest to newest; wall capture time is not simulation time.")
