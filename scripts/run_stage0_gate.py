import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sectvid.config import load_config
from sectvid.eval.report import write_stage0_gate

cfg = load_config()
p, gate = write_stage0_gate(cfg["report"]["dir"], cfg)
print(p)
print(json.dumps(gate, indent=1))
data = json.loads(Path(p).read_text())
ev = data["evidence"]["extractor_validation"]
print(json.dumps(ev, indent=1))
