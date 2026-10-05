"""Build the structure cache for every clip (spec 6.1, Stage 1).

Extraction uses the production path: VIDEO-mode trackers, EMA smoothing
(config structure.smoothing), per-frame camera affine. Resumable: clips with
an existing cache file are skipped. Progress lines go to stdout.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sectvid.config import load_config, root_path
from sectvid.data import clips as data_clips
from sectvid.structure import cache
from sectvid.structure.extractor import StructureExtractor


def main():
    cfg = load_config()
    ids = data_clips.discover_clips(cfg)
    t0 = time.time()
    done, skipped, failed, extract_time = 0, 0, [], 0.0
    ext = StructureExtractor(cfg)
    for i, cid in enumerate(ids, 1):
        try:
            cache.validate_structure(cfg, cid)
        except Exception:
            pass
        else:
            print(f"[{i}/{len(ids)}] {cid}: cached, skip", flush=True)
            skipped += 1
            continue
        props = data_clips.clip_properties(cfg, cid)
        frames = data_clips.load_frames(cfg, cid, 0, props["n_frames"])
        if not frames:
            print(f"[{i}/{len(ids)}] {cid}: no frames, skipping", flush=True)
            failed.append(cid)
            continue
        t1 = time.time()
        structs = ext.extract_sequence(frames, fps=props["fps"] or 24.0, smooth=True)
        cache.save_structure(cfg, cid, structs)
        ok_rate = float(sum(s.ok for s in structs)) / len(structs)
        elapsed = time.time() - t1
        extract_time += elapsed
        done += 1
        remaining = len(ids) - i
        eta = (extract_time / done) * remaining if done else 0.0
        print(f"[{i}/{len(ids)}] {cid}: {len(frames)} frames in {elapsed:.0f}s "
              f"(ok {ok_rate:.2f}), eta {eta/60:.0f} min", flush=True)
    print(f"done: extracted {done}, skipped {skipped}, failed {len(failed)} "
          f"({failed}), total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
