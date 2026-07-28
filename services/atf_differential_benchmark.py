from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from services.atf_class_refinement import WORKSPACE_ROOT, generate_tasks, run_task

RUNTIME_ROOT = Path("runtime/atf_differential")


REFERENCE: dict[str, str] = {
"dog_ethology_001": '''class Dog:
    def __init__(self, name, breed, age_years, energy=0.6, hunger=0.4):
        if age_years < 0: raise ValueError("age_years")
        self.name, self.breed, self.age_years = str(name), str(breed), float(age_years)
        self.energy, self.hunger = self._unit(energy), self._unit(hunger)
        self._update_mood()
    @staticmethod
    def _unit(value): return max(0.0, min(1.0, float(value)))
    def _update_mood(self):
        self.mood = "tired" if self.energy < .25 else "hungry" if self.hunger > .7 else "happy" if self.energy > .65 and self.hunger < .35 else "calm"
    def feed(self, amount):
        if amount < 0: raise ValueError("amount")
        self.hunger = self._unit(self.hunger - amount); self._update_mood(); return self.state()
    def play(self, minutes):
        if minutes < 0: raise ValueError("minutes")
        self.energy = self._unit(self.energy - minutes / 120); self.hunger = self._unit(self.hunger + minutes / 180); self._update_mood(); return self.state()
    def rest(self, hours):
        if hours < 0: raise ValueError("hours")
        self.energy = self._unit(self.energy + hours / 8); self._update_mood(); return self.state()
    def state(self): return {"name": self.name, "breed": self.breed, "age_years": self.age_years, "energy": self.energy, "hunger": self.hunger, "mood": self.mood}
''',
"bicycle_drivetrain_001": '''import math
class Bicycle:
    def __init__(self, wheel_radius_m, chainring_teeth, sprocket_teeth, rider_mass_kg):
        self.wheel_radius_m=float(wheel_radius_m); self.rider_mass_kg=float(rider_mass_kg)
        self.chainring_teeth=int(chainring_teeth); self.sprocket_teeth=int(sprocket_teeth); self._validate()
    def _validate(self):
        if min(self.wheel_radius_m,self.rider_mass_kg,self.chainring_teeth,self.sprocket_teeth)<=0: raise ValueError("positive values required")
    def gear_ratio(self): return self.chainring_teeth/self.sprocket_teeth
    def speed_mps(self, cadence_rpm):
        if cadence_rpm < 0: raise ValueError("cadence_rpm")
        return cadence_rpm/60*self.gear_ratio()*2*math.pi*self.wheel_radius_m
    def kinetic_energy_j(self, cadence_rpm): return .5*self.rider_mass_kg*self.speed_mps(cadence_rpm)**2
    def shift(self, chainring_teeth=None, sprocket_teeth=None):
        new_chainring=self.chainring_teeth if chainring_teeth is None else int(chainring_teeth)
        new_sprocket=self.sprocket_teeth if sprocket_teeth is None else int(sprocket_teeth)
        if new_chainring<=0 or new_sprocket<=0: raise ValueError("positive teeth required")
        self.chainring_teeth,self.sprocket_teeth=new_chainring,new_sprocket
''',
"avian_development_001": '''class BirdDevelopment:
    def __init__(self, species, age_days=0, mass_g=1.0):
        if age_days < 0 or mass_g <= 0: raise ValueError("invalid initial state")
        self.species, self.age_days, self.mass_g = str(species), int(age_days), float(mass_g); self.stage=self.stage_for_age(self.age_days)
    @staticmethod
    def stage_for_age(age_days):
        if age_days < 0: raise ValueError("age_days")
        return "egg" if age_days < 14 else "hatchling" if age_days < 28 else "nestling" if age_days < 45 else "fledgling" if age_days < 90 else "juvenile" if age_days < 180 else "adult"
    def advance(self, days, food_factor=1.0):
        if days < 0 or food_factor < 0: raise ValueError("non-negative inputs required")
        self.age_days += int(days); self.mass_g *= 1 + min(days, 365)*.015*float(food_factor); self.stage=self.stage_for_age(self.age_days); return self.state()
    def state(self): return {"species":self.species,"age_days":self.age_days,"mass_g":self.mass_g,"stage":self.stage}
''',
"thermal_reservoir_001": '''class ThermalReservoir:
    def __init__(self, mass_kg, specific_heat_j_per_kg_k, temperature_k):
        self.mass_kg=float(mass_kg); self.specific_heat_j_per_kg_k=float(specific_heat_j_per_kg_k); self.temperature_k=float(temperature_k)
        if self.mass_kg<=0 or self.specific_heat_j_per_kg_k<=0 or self.temperature_k<0: raise ValueError("invalid reservoir")
    def heat_capacity(self): return self.mass_kg*self.specific_heat_j_per_kg_k
    def add_heat(self, joules):
        value=self.temperature_k+float(joules)/self.heat_capacity()
        if value<0: raise ValueError("temperature below absolute zero")
        self.temperature_k=value; return value
    def energy_j(self): return self.heat_capacity()*self.temperature_k
    def mix(self, other):
        if not isinstance(other,ThermalReservoir): raise TypeError("other")
        capacity=self.heat_capacity()+other.heat_capacity(); temperature=(self.energy_j()+other.energy_j())/capacity
        return ThermalReservoir(self.mass_kg+other.mass_kg,capacity/(self.mass_kg+other.mass_kg),temperature)
''',
}

HIDDEN: dict[str, str] = {
"dog_ethology_001": '''from solution import Dog
d=Dog("x","y",1,2,-1); state=d.state(); assert state["energy"]==1 and state["hunger"]==0
for action,arg in ((d.feed,-1),(d.play,-1),(d.rest,-1)):
 try: action(arg)
 except ValueError: pass
 state=d.state(); assert 0<=state["energy"]<=1 and 0<=state["hunger"]<=1
assert set(d.state())=={"name","breed","age_years","energy","hunger","mood"}
''',
"bicycle_drivetrain_001": '''from solution import Bicycle
import math
b=Bicycle(.3,48,16,70); assert abs(b.speed_mps(60)-1.8*math.pi)<1e-9
for args in ((-1,),):
 try: b.speed_mps(*args); raise AssertionError("negative cadence accepted")
 except ValueError: pass
old=b.gear_ratio()
try: b.shift(sprocket_teeth=0); raise AssertionError("invalid shift")
except ValueError: pass
assert b.gear_ratio()==old
''',
"avian_development_001": '''from solution import BirdDevelopment
b=BirdDevelopment("x",13,2); assert b.state()["stage"]=="egg"; b.advance(1); assert b.state()["stage"]=="hatchling"
for args in ((-1,1),(1,-1)):
 try: b.advance(*args)
 except ValueError: pass
 state=b.state(); assert state["age_days"]>=0 and state["mass_g"]>0
assert b.stage_for_age(179)=="juvenile" and b.stage_for_age(180)=="adult"
''',
"thermal_reservoir_001": '''from solution import ThermalReservoir
a=ThermalReservoir(2,500,300); before=a.energy_j(); a.add_heat(-500); assert abs(a.energy_j()-(before-500))<1e-8
try: a.add_heat(-a.energy_j()-1); raise AssertionError("below zero accepted")
except ValueError: pass
b=ThermalReservoir(3,700,330); c=a.mix(b); assert abs(c.energy_j()-(a.energy_j()+b.energy_j()))<1e-7
''',
}


def _run(source: str, task_id: str) -> tuple[bool, float, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); (root/"solution.py").write_text(source,encoding="utf-8")
        script=HIDDEN[task_id]+"\nprint('PASS')\n"
        started=time.perf_counter()
        result=subprocess.run([sys.executable,"-c",script],cwd=root,capture_output=True,text=True,timeout=15)
        return result.returncode==0, time.perf_counter()-started, (result.stderr or result.stdout)[-1200:]


def _static_metrics(source: str) -> dict[str, Any]:
    tree=ast.parse(source)
    return {"lines":len([line for line in source.splitlines() if line.strip()]),"ast_nodes":sum(1 for _ in ast.walk(tree)),"bare_except":sum(isinstance(n,ast.ExceptHandler) and n.type is None for n in ast.walk(tree))}


def evaluate(task_id: str, candidate: str) -> dict[str, Any]:
    reference=REFERENCE[task_id]
    ref_ok,ref_time,ref_error=_run(reference,task_id)
    try: cand_ok,cand_time,cand_error=_run(candidate,task_id); cand_metrics=_static_metrics(candidate)
    except Exception as exc: cand_ok,cand_time,cand_error,cand_metrics=False,0.0,str(exc),{"lines":0,"ast_nodes":0,"bare_except":99}
    ref_metrics=_static_metrics(reference)
    efficiency_ratio=cand_time/max(ref_time,1e-9)
    complexity_ratio=cand_metrics["ast_nodes"]/max(ref_metrics["ast_nodes"],1)
    score=(70 if cand_ok else 0)+(10 if efficiency_ratio<=3 else max(0,10-(efficiency_ratio-3)*2))+(10 if complexity_ratio<=1.75 else max(0,10-(complexity_ratio-1.75)*4))+(10 if cand_metrics["bare_except"]==0 else 0)
    return {"task_id":task_id,"candidate_hidden_passed":cand_ok,"reference_hidden_passed":ref_ok,"score":round(score,2),"parity":bool(cand_ok and score>=95),"runtime_ratio":round(efficiency_ratio,3),"complexity_ratio":round(complexity_ratio,3),"candidate_metrics":cand_metrics,"reference_metrics":ref_metrics,"candidate_error":cand_error,"reference_error":ref_error}


def run_differential(*, attempts: int=2, regenerate: bool=True) -> dict[str, Any]:
    RUNTIME_ROOT.mkdir(parents=True,exist_ok=True); results=[]
    for task in generate_tasks(4):
        if regenerate: generation=run_task(task,attempts=attempts)
        else: generation={"status":"existing"}
        path=WORKSPACE_ROOT/task.task_id/"solution.py"
        result=evaluate(task.task_id,path.read_text(encoding="utf-8") if path.exists() else "")
        repairs=[]
        for repair_attempt in range(1, 3):
            if result["parity"] or not regenerate:
                break
            repair=_repair_hidden_failure(task,root=path.parent,failure=result["candidate_error"],attempt=repair_attempt)
            repairs.append(repair)
            result=evaluate(task.task_id,path.read_text(encoding="utf-8") if path.exists() else "")
        result["hidden_repairs"]=repairs
        result["generation"]=generation; results.append(result)
    payload={"created_at":time.time(),"parity_count":sum(r["parity"] for r in results),"count":len(results),"average_score":round(sum(r["score"] for r in results)/len(results),2),"results":results}
    (RUNTIME_ROOT/"latest.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return payload


def _repair_hidden_failure(task: Any, *, root: Path, failure: str, attempt: int) -> dict[str, Any]:
    from tools.c0d3rV2.delivery_runner import run_delivery_turn_detailed
    prompt=(
        f"Update solution.py for the public class {task.class_name} described by contract.json. "
        "Do not change test_solution.py. Run: test_solution.py. This is one class and one file. "
        "Preserve all passing behavior. An independent hidden acceptance check observed this failure:\n"
        f"{failure[-2500:]}\nMake the smallest contract-consistent correction, then run test_solution.py."
    )
    started=time.perf_counter()
    try:
        detail=run_delivery_turn_detailed(
            prompt,session_key=f"atf-differential:{task.task_id}:hidden:{attempt}",workdir=root,
            backend="freeloader",system_context="Validator-driven differential repair; observed evidence is authoritative and scope is locked.",reset=True,
        )
        return {"attempt":attempt,"ok":True,"duration_sec":round(time.perf_counter()-started,3),"models":detail.get("models",[]),"model_calls":detail.get("turn_model_calls"),"output":str(detail.get("output") or "")[:1000]}
    except Exception as exc:
        return {"attempt":attempt,"ok":False,"duration_sec":round(time.perf_counter()-started,3),"error":str(exc)[:1500]}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--existing",action="store_true"); parser.add_argument("--attempts",type=int,default=2); args=parser.parse_args()
    print(json.dumps(run_differential(attempts=args.attempts,regenerate=not args.existing),indent=2))
