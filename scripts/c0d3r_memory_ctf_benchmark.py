from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
for item in (ROOT,ROOT/"tools",ROOT/"tools"/"c0d3rV2"):
    if str(item) not in sys.path: sys.path.insert(0,str(item))

from tools.c0d3rV2.lt_mem import LongTermMemory
from tools.c0d3rV2.st_memory import STMemory
from tools.c0d3rV2.side_load_st_mem_file_location import STSideLoadedMemory
from tools.c0d3rV2.side_load_lt_mem_file_location import LTSideLoadedMemory
from tools.c0d3rV2.plugins.agent_the_freeloader import AgentTheFreeloaderSession

RUNTIME=ROOT/"runtime"/"memory_ctf"
ARENA=RUNTIME/"arena"


@dataclass
class AgentResult:
    team:str; captured:bool; turns:int; errors:list[str]; events:list[dict[str,Any]]


class ScopedArenaOps:
    def __init__(self,team:str,opponent:str,st:STSideLoadedMemory,lt:LTSideLoadedMemory):
        self.team=team;self.opponent=opponent;self.home=(ARENA/team).resolve();self.st=st;self.lt=lt
    def execute(self,action:dict[str,Any])->dict[str,Any]:
        kind=str(action.get("action") or "").lower(); raw=str(action.get("path") or action.get("source") or "")
        if kind=="search":
            query=str(action.get("query") or "flag clue")
            remembered=self.st.lookup(query,cwd=str(ARENA))+self.lt.lookup(query,cwd=str(ARENA),project_root=str(ARENA))
            tokens=[t for t in re.findall(r"[a-z0-9_-]+",query.lower()) if len(t)>2]
            live=[]
            for path in ARENA.rglob("*"):
                if path.is_file() and (not tokens or any(t in path.name.lower() or t in str(path.parent).lower() for t in tokens)):
                    live.append(str(path.resolve()))
            paths=list(dict.fromkeys([p for p in remembered if Path(p).exists()]+live))[:20]
            self.st.record_paths(query,paths,cwd=str(ARENA),project_root=str(ARENA));self.lt.record_paths(query,paths,cwd=str(ARENA),project_root=str(ARENA),session_id=self.st.session_id)
            return {"ok":True,"paths":paths}
        path=self._arena_path(raw)
        if kind=="read":
            if not path.is_file(): return {"error":"file_not_found"}
            self.st.record_paths(str(action.get("purpose") or path.name),[str(path)],cwd=str(ARENA),project_root=str(ARENA))
            return {"ok":True,"path":str(path),"content":path.read_text(encoding="utf-8",errors="replace")[:5000]}
        if kind=="write":
            self._require_home(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(str(action.get("content") or ""),encoding="utf-8");self.st.record_paths("written file",[str(path)],cwd=str(ARENA),project_root=str(ARENA));return {"ok":True,"path":str(path)}
        if kind in {"copy","move"}:
            source=self._arena_path(str(action.get("source") or ""));dest=self._arena_path(str(action.get("destination") or ""));self._require_home(dest)
            if not source.is_file(): return {"error":"source_not_found"}
            dest.parent.mkdir(parents=True,exist_ok=True)
            if kind=="copy": shutil.copy2(source,dest)
            else:
                self._require_home(source);shutil.move(str(source),str(dest))
            self.st.record_paths(f"{kind} result",[str(source),str(dest)],cwd=str(ARENA),project_root=str(ARENA));return {"ok":True,"source":str(source),"path":str(dest)}
        return {"error":"unsupported_action","allowed":["search","read","write","copy","move"]}
    def _arena_path(self,raw:str)->Path:
        candidate=(ARENA/raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        candidate.relative_to(ARENA.resolve());return candidate
    def _require_home(self,path:Path)->None:path.relative_to(self.home)


class MemoryCtfAgent:
    def __init__(self,team:str,opponent:str):
        self.team=team;self.opponent=opponent;self.session_id=f"ctf-{team}"
        self.session=AgentTheFreeloaderSession(session_name=self.session_id,transcript_dir=RUNTIME/"transcripts",workdir=ARENA,timeout_s=20,max_attempts=4,max_tokens=900)
        self.lt=LongTermMemory(RUNTIME);self.short=STMemory(self.session,session_id=self.session_id,runtime_root=RUNTIME)
        self.st=STSideLoadedMemory(self.session_id,RUNTIME);self.lt_side=LTSideLoadedMemory(RUNTIME);self.ops=ScopedArenaOps(team,opponent,self.st,self.lt_side)
        self.events=[];self.errors=[]
    def turn(self,instruction:str)->dict[str,Any]:
        recall=self.lt.search(instruction,limit=8)
        hazy=self.st.lookup_detailed("opponent flag clue",cwd=str(ARENA))+self.lt_side.lookup_detailed("opponent flag clue",cwd=str(ARENA),project_root=str(ARENA))
        required=["search","read","write","move","copy"]
        acceptance=self._acceptance_state()
        completed={kind for kind,done in acceptance.items() if done}
        next_required=next((kind for kind in required if kind not in completed),"read")
        contracts={
            "search":{"action":"search","query":f"{self.opponent}/vault/flag.txt current arena"},
            "read":{"action":"read","path":str((ARENA/self.opponent/"vault"/"flag.txt").resolve())},
            "write":{"action":"write","path":str((ARENA/self.team/"notes"/"plan.txt").resolve()),"content":"A useful current-game plan of at least 15 characters"},
            "move":{"action":"move","source":str((ARENA/self.team/"inbox"/"decoy.txt").resolve()),"destination":str((ARENA/self.team/"archive"/"decoy.txt").resolve())},
            "copy":{"action":"copy","source":str((ARENA/self.opponent/"vault"/"flag.txt").resolve()),"destination":str((ARENA/self.team/"captures"/f"{self.opponent}.flag").resolve())},
        }
        prompt={"role":f"CTF agent {self.team}","opponent":self.opponent,"objective":f"Complete every checklist item: (1) search for the current mission/flag, (2) read the clue or opponent flag, (3) write a useful plan to {self.team}/notes/plan.txt, (4) move {self.team}/inbox/decoy.txt to {self.team}/archive/decoy.txt, and (5) copy the exact {self.opponent} flag into {self.team}/captures/{self.opponent}.flag. Never write outside your own team directory.","instruction":instruction,"short_memory":self.short.summary_bundle,"recent_transcript":self.short.build_transcript_section(3500),"long_term_recall":recall,"hazy_hash_candidates":hazy[:10],"acceptance_postconditions":acceptance,"required_next_action_contract":contracts[next_required],"last_events":self.events[-5:],"action_schema":{"action":"search|read|write|copy|move","query":"for search","path":"for read/write","content":"for write","source":"for copy/move","destination":"for copy/move"}}
        try:
            raw=self.session.send(json.dumps(prompt,default=str),system=f"You are one autonomous C0d3rV2 capture-the-flag filesystem agent. Return exactly one JSON action implementing required_next_action_contract. Preserve its action, source/path, and destination exactly; only author useful content when action=write. Memory paths are hypotheses, not current filesystem truth. No prose or markdown.",temperature=.1,max_tokens=900)
            action=_json_object(raw);result=self.ops.execute(action)
        except Exception as exc:
            action={};result={"error":str(exc)}
        event={"ts":time.time(),"action":action,"result":result};self.events.append(event)
        if result.get("error"):self.errors.append(str(result["error"]))
        summary=f"Action {json.dumps(action,default=str)} produced {json.dumps(result,default=str)[:1500]}"
        self.short.record_turn(instruction,summary,has_error=bool(result.get("error")),has_tool_calls=True,update_summary_model=False)
        self.lt.append(instruction,summary,workdir=str(ARENA),model_id=self.session.get_model_id(),session_id=self.session_id)
        return event
    def captured(self)->bool:
        return all(self._acceptance_state().values())
    def _acceptance_state(self)->dict[str,bool]:
        expected=(ARENA/self.opponent/"vault"/"flag.txt").read_text(encoding="utf-8")
        captured=ARENA/self.team/"captures"/f"{self.opponent}.flag";plan=ARENA/self.team/"notes"/"plan.txt";archive=ARENA/self.team/"archive"/"decoy.txt";inbox=ARENA/self.team/"inbox"/"decoy.txt"
        good=[event for event in self.events if not event.get("result",{}).get("error")]
        return {
            "search":any(event.get("action",{}).get("action")=="search" and event.get("result",{}).get("paths") for event in good),
            "read":any(event.get("action",{}).get("action")=="read" and str(event.get("result",{}).get("content") or "")==expected for event in good),
            "write":plan.is_file() and len(plan.read_text(encoding="utf-8",errors="ignore"))>=15,
            "move":archive.is_file() and not inbox.exists(),
            "copy":captured.is_file() and captured.read_text(encoding="utf-8",errors="ignore")==expected and any(event.get("action",{}).get("action")=="copy" for event in good),
        }
    def close(self)->None:self.lt_side.absorb_from_session(self.st.hazy_hash)


def _json_object(raw:str)->dict[str,Any]:
    start,end=str(raw).find("{"),str(raw).rfind("}")
    if start<0 or end<=start:raise ValueError("model_did_not_return_json_action")
    value=json.loads(str(raw)[start:end+1]);return value if isinstance(value,dict) else {}


def setup_arena()->None:
    shutil.rmtree(ARENA,ignore_errors=True)
    for team,opponent in (("red","blue"),("blue","red")):
        (ARENA/team/"vault").mkdir(parents=True);(ARENA/team/"notes").mkdir();(ARENA/team/"captures").mkdir()
        (ARENA/team/"inbox").mkdir();(ARENA/team/"archive").mkdir()
        (ARENA/team/"inbox"/"decoy.txt").write_text(f"{team} owned decoy",encoding="utf-8")
        (ARENA/team/"vault"/"flag.txt").write_text(f"FLAG{{{team}-{uuid.uuid4().hex[:12]}}}",encoding="utf-8")
        clue=ARENA/"shared"/f"clue-for-{team}.txt";clue.parent.mkdir(exist_ok=True)
        clue.write_text(f"Agent {team}: the opponent is {opponent}. Search the arena for {opponent}/vault/flag.txt, read it, then copy that observed file to {team}/captures/{opponent}.flag.",encoding="utf-8")


def read_only_seed_common_roots(lt:LTSideLoadedMemory,limit:int=600)->int:
    roots=[ROOT,Path.home()/"Desktop",Path.home()/"Documents",Path("D:/Projects")];paths=[]
    ignored={".git","node_modules",".venv","venv","__pycache__"}
    for root in roots:
        if not root.exists():continue
        try:
            for path in root.rglob("*"):
                if len(paths)>=limit:break
                if path.is_file() and not any(part in ignored for part in path.parts):paths.append(str(path.resolve()))
        except (OSError,PermissionError):continue
    lt.record_paths("read-only computer file index",paths,cwd=str(ROOT),project_root=str(ROOT));return len(paths)


def run(max_turns:int=8)->dict[str,Any]:
    setup_arena();seed=LTSideLoadedMemory(RUNTIME);indexed=read_only_seed_common_roots(seed)
    agents={name:MemoryCtfAgent(name,"blue" if name=="red" else "red") for name in ("red","blue")}
    # Seed only each team's clue into its session Hazy Hash; flags remain undisclosed.
    for name,agent in agents.items():agent.st.record_paths("initial mission clue",[str((ARENA/"shared"/f"clue-for-{name}.txt").resolve())],cwd=str(ARENA),project_root=str(ARENA))
    for turn in range(1,max_turns+1):
        for name in ("red","blue"):
            agent=agents[name]
            if not agent.captured():agent.turn("Do you remember when we started this CTF project? Continue from memory and take exactly one useful filesystem action toward capturing the opponent flag.")
        if turn==2:
            # Hard restart proves ST disk persistence, LT SQLite recall, and LT Hazy promotion.
            saved={name:(agent.events,agent.errors) for name,agent in agents.items()}
            for agent in agents.values():agent.close()
            agents={name:MemoryCtfAgent(name,"blue" if name=="red" else "red") for name in ("red","blue")}
            for name,agent in agents.items():agent.events,agent.errors=saved[name]
        if all(agent.captured() for agent in agents.values()):break
    results={name:AgentResult(name,agent.captured(),len(agent.events),agent.errors,agent.events).__dict__ for name,agent in agents.items()}
    payload={"created_at":time.time(),"arena":str(ARENA),"read_only_paths_indexed":indexed,"session_restart_after_turn":2,"passed":all(item["captured"] for item in results.values()),"agents":results}
    RUNTIME.mkdir(parents=True,exist_ok=True);(RUNTIME/"latest.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8");return payload


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--max-turns",type=int,default=8);args=parser.parse_args();print(json.dumps(run(args.max_turns),indent=2,default=str))
