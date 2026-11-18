"""End-to-end test of scoring + retraining on SIMULATED sessions, in a throwaway state directory.

Run:  python ml/service/selftest.py <empty-directory>
"""
import json
import os
import sys
import time
from pathlib import Path

state = Path(sys.argv[1]).resolve()
state.mkdir(parents=True, exist_ok=True)
os.environ["ML_STATE_DIR"] = str(state)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine as E  # noqa: E402
from retrain import retrain  # noqa: E402
from simulate_sessions import simulate  # noqa: E402

t0 = time.time()
sessions = simulate(160)
print(f"simulated {len(sessions)} sessions in {time.time() - t0:.0f}s")

print("\n1) score one session with the starting registry")
print(json.dumps(E.score_session(sessions[0]), indent=1)[:1800])

print("\n2) retrain from labelled sessions")
t0 = time.time()
rep = retrain(sessions[:120])
for d, r in rep["results"].items():
    print(d, {k: r.get(k) for k in ["status", "labelled_sessions", "at_risk", "challenger_model", "challenger_roc_auc",
                                     "champion", "champion_roc_auc", "champion_score_basis", "champion_unseen_sessions"]})
print(f"retrain took {time.time() - t0:.0f}s")

print("\n3) score an unseen session with whatever is active now")
res = E.score_session(sessions[150])
print({d: (v.get("status"), v.get("model_version"), round(v["probability"], 3) if "probability" in v else None,
           v.get("risk_level")) for d, v in res["domains"].items()})

print("\n4) retrain again with more data (champion now a game model for some disorders)")
rep2 = retrain(sessions)
for d, r in rep2["results"].items():
    print(d, {k: r.get(k) for k in ["status", "challenger", "challenger_roc_auc", "champion",
                                     "champion_roc_auc", "champion_score_basis", "champion_unseen_sessions"]})
reg = E.load_registry()
print("\nactive:", {d: reg[d]["active"] for d in E.DISORDERS})
