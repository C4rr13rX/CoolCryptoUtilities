import os, sys, json, time
from pathlib import Path
ROOT=Path(".").resolve(); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"web")); sys.path.insert(0,str(ROOT/"tools"/"c0d3rV2"))
os.environ.setdefault("DJANGO_SETTINGS_MODULE","coolcrypto_dashboard.settings"); os.environ["SECURE_ENV_HYDRATED"]="1"
import django; django.setup()
from services.env_loader import EnvLoader; EnvLoader.load()

WORK=Path(r"C:\Users\Adam\Desktop\Multiscale Robot World")
next_step="Set up TypeScript/Three.js project foundation with build tooling and folder structure"
exec_prompt=("You are the solo delivery agent. Execute the next step in the repo. "
  "Follow the existing project conventions, run a quick smoke test when done, and summarize what changed. If blocked, explain why.\n"
  f"Current plan summary: multiscale robot world\nNext step: {next_step}\n")
atomic_contract=(f"Atomic work package: {next_step}\n"
  "Make concrete progress for this package, validate it, and stop.\n"
  'Expected outputs: ["package.json", "src/", "tests/"]\n')
sys_ctx=(f"Project: Multiscale Robot World\nWorkdir: {WORK}\nRun ID: repro\n"
  "This is an unattended atomic workday job. Execute only the named work package, use bounded context, perform changes/tests when requested, and return a user-facing result.")

# count files before
def count(): return sum(1 for p in WORK.rglob("*") if p.is_file() and ".git" not in p.parts)
before=count()
print("files before:",before)

from tools.ai_backend_mode import freeloader_mode_active
import tools.c0d3rV2.delivery_runner as dr
flow=dr._build_delivery_flow("repro-step4", WORK, backend="freeloader")
print("session class:", type(flow.session).__name__)
begin=getattr(flow.session,"begin_turn",None)
if callable(begin): begin(int(os.getenv("C0D3R_MAX_MODEL_CALLS","40")))
flow._pending_system=sys_ctx.strip(); dr._patch_session_context(flow, sys_ctx)
prompt=exec_prompt+"\n"+atomic_contract

# atomic path check
from outline_refiner import OutlineRefiner
print("contract_ready:", OutlineRefiner._contract_ready(prompt))

aug=flow.step_2_inject_context(prompt)
from orchestrator import Orchestrator
from petal_system import PetalManager
orch=Orchestrator(session=flow.session, tools=flow.tools, context=flow._context, petals=flow.petals or PetalManager())
t0=time.time()
try:
    results, tree = orch.run(prompt)
    print("orch.run OK in", round(time.time()-t0,1),"s")
except Exception as e:
    import traceback; print("ORCH EXCEPTION:", repr(e)); traceback.print_exc(); results,tree=[],None

print("\n=== RESULTS ===")
for r in (results or []):
    out=(getattr(r,'output','') or '')
    print(" - output_chars=",len(out),"has_tool_outputs=",bool(getattr(r,'tool_outputs',None)),"| ",out[:150].replace("\n"," "))

print("\n=== TOOL EVENTS ===")
evs=list(tree.accumulated_results()) if tree is not None else []
for e in evs:
    if isinstance(e,dict) and e.get("tool"):
        res=e.get("result") or {}
        print(f" - {e.get('tool')}: error={ (str(res.get('error'))[:80] if res.get('error') else None) } keys={list(res.keys())[:6]}")
after=count()
print("\nfiles after:",after,"(delta",after-before,")")
