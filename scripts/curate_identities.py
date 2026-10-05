"""Identity curation (spec 6.1 prerequisite for splits).

Clips default to identity_id = clip_id, which silently assumes one person per
clip. Splits are identity-first, so that assumption must be human-verified
before split generation. This script:

  --propose   group clips by source-video prefix and seed
              data/meta/identities.json with one candidate identity per source
              group (for owner review: merge groups that share a person, split
              groups that contain several, rename identities, drop clips).
  --apply     read data/meta/identities.json and write identity_id +
              identity_reviewed=true into each listed clip's meta.
  --check     report reviewed vs unreviewed clips.

The curation file format (data/meta/identities.json):
  {"identities": {"<identity_id>": {"clips": ["clip_a", ...], "note": "..."}}}
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sectvid.config import load_config
from sectvid.data import clips as data_clips
from sectvid.data import splits

_SUFFIX = re.compile(r"^(?P<src>.*?)(?:_clip_\d+(?:_c\d+)?|_c\d+)?$")


def source_prefix(clip_id):
    m = _SUFFIX.fullmatch(clip_id)
    if m and m.group("src"):
        return m.group("src")
    return clip_id


def curation_path(cfg):
    return data_clips.meta_dir(cfg) / "identities.json"


def slugify(s):
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s[:40] or "src"


def cmd_propose(cfg, force):
    path = curation_path(cfg)
    if path.exists() and not force:
        print(f"{path} already exists; use --force to reseed.")
        return 0
    ids = data_clips.discover_clips(cfg)
    groups = {}
    for cid in ids:
        groups.setdefault(source_prefix(cid), []).append(cid)
    identities = {}
    for src, cids in sorted(groups.items()):
        identities[f"src_{slugify(src)}"] = {"clips": sorted(cids), "note": f"source: {src}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"identities": identities}, indent=1), encoding="utf-8")
    print(f"{len(groups)} source groups -> {path}")
    for ident, info in sorted(identities.items(), key=lambda kv: -len(kv[1]["clips"])):
        print(f"  {ident}  ({len(info['clips'])} clips)  {info['note']}")
    print("Review: merge groups that share a person, split groups with several,")
    print("rename as needed, then run: python scripts/curate_identities.py --apply")
    return 0


def cmd_apply(cfg):
    path = curation_path(cfg)
    if not path.exists():
        print(f"no curation file at {path}; run --propose first.")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    identities = data.get("identities", {})
    covered = set()
    for ident, info in identities.items():
        for cid in info.get("clips", []):
            covered.add(cid)
            data_clips.set_meta(cfg, cid, {"identity_id": ident, "identity_reviewed": True})
    all_ids = data_clips.discover_clips(cfg)
    metas = {cid: data_clips.load_meta(cfg, cid) for cid in all_ids}
    unreviewed = splits.curation_check(metas)
    print(f"applied {len(identities)} identities to {len(covered)} clips")
    print(f"unreviewed: {len(unreviewed)} of {len(all_ids)}")
    for cid in unreviewed:
        print(f"  {cid}")
    return 0 if not unreviewed else 2


def cmd_check(cfg):
    all_ids = data_clips.discover_clips(cfg)
    metas = {cid: data_clips.normalized_meta(cfg, cid) for cid in all_ids}
    unreviewed = splits.curation_check(metas)
    print(f"reviewed: {len(all_ids) - len(unreviewed)} / {len(all_ids)}")
    if unreviewed:
        print(f"unreviewed: {len(unreviewed)}")
        for cid in unreviewed:
            print(f"  {cid}")
    return 0 if not unreviewed else 2


def main():
    cfg = load_config()
    args = sys.argv[1:]
    if "--propose" in args:
        return cmd_propose(cfg, "--force" in args)
    if "--apply" in args:
        return cmd_apply(cfg)
    if "--check" in args:
        return cmd_check(cfg)
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
