"""Paired language/tool attack for temporal validity.

Each case is a timestamped conversation with one cached fact and one function,
refresh_source. The model must either answer from cache or call the function.

Conditions (same hidden episode/history/tool):
  raw       timestamps + source validity contract
  age_plane raw + deterministic (world age, arrival age, intervening messages,
            invalidation); no recommendation
  resolver  raw + deterministic contract decision

Sources:
  weather      valid through 8 s after observed_at
  discourse    valid through 8 intervening messages after cache receipt
  reservation  valid until an explicit change notification

Commands:
  generate, sanity, run-openai, score
"""
from __future__ import annotations

import argparse, concurrent.futures, json, math, os, random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import numpy as np

SOURCES=("weather","discourse","reservation")
CONDS=("raw","age_plane","resolver")
TTL_S=8.0; TTL_N=8
R_GOOD=1.0; R_BAD=-1.5; R_REFRESH=.55
REGIMES={
 "iid":(.95,1.05,1.5), "dense":(.25,.45,1.5), "sparse":(1.8,2.4,1.5),
 "long_delay":(.95,1.05,7.0), "dense_long":(.25,.45,7.0),
}
PLACES=("Northport","Lakeview","Riverton","Hillcrest","Westhaven")
WEATHER=("light rain, 9 C","clear, 14 C","overcast, 11 C","windy, 8 C","drizzle, 7 C")
PROJECTS=("Project Alder","Project Birch","Project Cedar","Project Delta","Project Elm")
STATES=("CONFIRMED","HELD","PAID","CHECKED-IN")
TOOL={"type":"function","name":"refresh_source","description":"Fetch a fresh current value only when the cached evidence is invalid under the contract.","parameters":{"type":"object","properties":{"source":{"type":"string","enum":list(SOURCES)},"key":{"type":"string"}},"required":["source","key"],"additionalProperties":False},"strict":True}
CONTRACT="""You are a cache-aware tool agent. For the final user request, either answer directly from the cached evidence or call refresh_source. Follow this validity contract exactly:
- WEATHER: valid through 8.0 seconds after observed_at. Use observed_at, not received_at.
- DISCOURSE: valid through 8 intervening conversation messages after the cache record was received. Wall-clock seconds do not expire it.
- RESERVATION: does not expire with time/message count; invalid only after an explicit reservation-change notification after the cache.
If valid, answer directly and do not call the tool. If invalid, call refresh_source and do not guess. The final user message is the decision point and is not an intervening message."""

def iso(t): return t.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00","Z")
def readjl(p):
    with Path(p).open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]
def writejl(p, rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
def reward(a,o): return R_REFRESH if a=="refresh" else (R_GOOD if o=="reuse" else R_BAD)

def make_case(seed,i,regime):
    ridx=list(REGIMES).index(regime); rng=np.random.default_rng(seed+100003*i+17*ridx); pr=random.Random(seed+900001*i+31*ridx)
    source=SOURCES[int(rng.integers(3))]; n=int(rng.integers(0,17)); lo,hi,ds=REGIMES[regime]
    gap=float(rng.uniform(lo,hi)); gaps=gap*rng.uniform(.9,1.1,size=n); delay=min(float(rng.exponential(ds)),16.0 if source=="weather" else 8.0)
    received=datetime(2026,1,15,12,0,tzinfo=timezone.utc)+timedelta(days=i//200,minutes=(i%200)*3)
    observed=received-timedelta(seconds=delay); current=received
    inval=False; slot=None
    if source=="reservation" and n:
        inval=bool(rng.random() < 1-(1-.075)**n); slot=int(rng.integers(n)) if inval else None
    if source=="weather":
        key=pr.choice(PLACES); value=pr.choice(WEATHER); cache=f"Cached weather record for {key}: {value}. observed_at={iso(observed)}; received_at={iso(received)}."; final=f"What is the weather in {key} now? Use cache if valid; otherwise refresh it."
    elif source=="discourse":
        key="current_draft_reference"; value=pr.choice(PROJECTS); cache=f"Cached conversation-focus resolution: 'the draft' refers to {value}. observed_at={iso(observed)}; received_at={iso(received)}."; final="What does 'the draft' refer to now? Use the cached resolution if valid; otherwise refresh it."
    else:
        key=f"R-{1000+i%8000:04d}"; value=pr.choice(STATES); cache=f"Cached reservation record: {key} status is {value}. observed_at={iso(observed)}; received_at={iso(received)}."; final=f"What is the current status of reservation {key}? Use cache if valid; otherwise refresh it."
    hist=[{"role":"assistant","time":iso(received),"content":cache}]
    for j in range(n):
        current += timedelta(seconds=float(gaps[j])); role="user" if j%2==0 else "assistant"
        if source=="reservation" and slot==j: text=f"Reservation-change notification: {key} changed externally after the cached record. The cached reservation status is no longer current."
        elif role=="user": text=f"Side note {j+1}: record neutral marker {int(rng.integers(100,999))}. This does not update any cached source."
        else: text=f"Marker {int(rng.integers(100,999))} recorded. No cached source was updated."
        hist.append({"role":role,"time":iso(current),"content":text})
    current += timedelta(seconds=gap*float(rng.uniform(.9,1.1)))
    decision=iso(current); world_age=(current-observed).total_seconds(); arrival_age=(current-received).total_seconds()
    hist.append({"role":"user","time":decision,"content":final})
    valid=(world_age<=TTL_S+1e-9) if source=="weather" else ((n<=TTL_N) if source=="discourse" else not inval)
    case={"case_id":f"{regime}-{i:05d}","regime":regime,"source":source,"key":key,"cached_value":value,"observed_at":iso(observed),"received_at":iso(received),"decision_at":decision,"valid_age_seconds":world_age,"arrival_age_seconds":arrival_age,"intervening_messages":n,"explicit_invalidation":inval,"oracle_action":"reuse" if valid else "refresh","history":hist}
    case["conditions"]={c:{"messages":render(case,c),"tools":[TOOL]} for c in CONDS}
    return case

def extra(case,c):
    if c=="raw": return ""
    if c=="age_plane": return f"\n\nTEMPORAL RUNTIME STATE (arithmetic only; no recommendation):\n- source: {case['source']}\n- current decision time: {case['decision_at']}\n- valid/world age: {case['valid_age_seconds']:.3f} seconds\n- arrival/knowledge age: {case['arrival_age_seconds']:.3f} seconds\n- intervening conversation messages: {case['intervening_messages']}\n- explicit invalidation after cache: {'yes' if case['explicit_invalidation'] else 'no'}\nApply the contract yourself."
    return f"\n\nDETERMINISTIC VALIDITY RESOLVER:\n- source: {case['source']}\n- cache_valid_under_contract: {'yes' if case['oracle_action']=='reuse' else 'no'}\n- recommended action: {case['oracle_action'].upper()}\nThe resolver applied the contract mechanically."
def render(case,c):
    out=[{"role":"system","content":CONTRACT+extra(case,c)}]
    out += [{"role":m["role"],"content":f"[{m['time']}] {m['content']}"} for m in case["history"]]
    return out

def cmd_generate(a):
    rows=[make_case(a.seed,i,r) for r in REGIMES for i in range(a.per_regime)]; writejl(a.output,rows); print(f"wrote {len(rows)} paired cases; {len(rows)*len(CONDS)} model decisions")

def meta_action(c,p):
    inv=c["explicit_invalidation"]
    if p=="arrival": ok=c["arrival_age_seconds"]<=TTL_S and not inv
    elif p=="timestamp": ok=c["valid_age_seconds"]<=TTL_S and not inv
    elif p=="position": ok=c["intervening_messages"]<=TTL_N and not inv
    elif p=="resolver": return c["oracle_action"]
    elif p=="always_reuse": return "reuse"
    else: return "refresh"
    return "reuse" if ok else "refresh"
def summary(rows):
    rows=[r for r in rows if r.get("action") in ("reuse","refresh")]
    if not rows: return (math.nan,)*4
    agree=np.mean([r["action"]==r["oracle_action"] for r in rows]); util=np.mean([reward(r["action"],r["oracle_action"]) for r in rows]); bad=np.mean([r["action"]=="reuse" and r["oracle_action"]=="refresh" for r in rows]); ref=np.mean([r["action"]=="refresh" for r in rows]); return agree,util,bad,ref

def cmd_sanity(a):
    cases=readjl(a.input)
    for c in cases:
        tail=c["conditions"]["raw"]["messages"][1:]
        assert c["conditions"]["age_plane"]["messages"][1:]==tail and c["conditions"]["resolver"]["messages"][1:]==tail
        assert c["conditions"]["raw"]["tools"]==c["conditions"]["age_plane"]["tools"]==c["conditions"]["resolver"]["tools"]
    print(f"pairing checks passed for {len(cases)} cases")
    for reg in REGIMES:
        sub=[c for c in cases if c["regime"]==reg]; print(f"\n{reg} n={len(sub)}")
        for p in ("arrival","timestamp","position","resolver","always_reuse","always_refresh"):
            ar=[{**c,"action":meta_action(c,p)} for c in sub]; ag,u,b,rr=summary(ar); print(f"  {p:>14s}: agreement={ag:.3f} utility={u:.3f} bad_reuse={b:.3f} refresh={rr:.3f}")

def response_action(resp):
    calls=[]
    for x in getattr(resp,"output",[]) or []:
        if getattr(x,"type",None)=="function_call": calls.append({"name":getattr(x,"name",None),"arguments":getattr(x,"arguments",None),"call_id":getattr(x,"call_id",None)})
    return ("refresh" if any(x["name"]=="refresh_source" for x in calls) else "reuse",getattr(resp,"output_text",None),calls)
def cmd_run(a):
    if not os.environ.get("OPENAI_API_KEY"): raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI
    cases=readjl(a.input); jobs=[(c,k) for c in cases for k in a.conditions]; jobs=jobs[:a.limit] if a.limit else jobs; client=OpenAI(base_url=a.base_url) if a.base_url else OpenAI()
    def one(job):
        c,k=job
        try:
            z=c["conditions"][k]; resp=client.responses.create(model=a.model,input=z["messages"],tools=z["tools"],tool_choice="auto"); act,text,calls=response_action(resp); err=None; rid=getattr(resp,"id",None)
        except Exception as e: act=text=rid=None; calls=[]; err=f"{type(e).__name__}: {e}"
        return {"case_id":c["case_id"],"regime":c["regime"],"source":c["source"],"condition":k,"model":a.model,"action":act,"output_text":text,"tool_calls":calls,"response_id":rid,"error":err}
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for j,r in enumerate(ex.map(one,jobs),1): rows.append(r); print(f"completed {j}/{len(jobs)}") if j%25==0 or j==len(jobs) else None
    writejl(a.output,rows)

def args_ok(row,c):
    for x in row.get("tool_calls") or []:
        if x.get("name")!="refresh_source": continue
        try: d=json.loads(x.get("arguments")) if isinstance(x.get("arguments"),str) else (x.get("arguments") or {})
        except Exception: continue
        if d.get("source")==c["source"] and str(d.get("key"))==str(c["key"]): return True
    return False
def task_ok(row,c):
    if c["oracle_action"]=="refresh": return row.get("action")=="refresh" and args_ok(row,c)
    return row.get("action")=="reuse" and str(c["cached_value"]).casefold() in (row.get("output_text") or "").casefold()
def cmd_score(a):
    cases={c["case_id"]:c for c in readjl(a.cases)}; rows=[]
    for r in readjl(a.responses):
        if r["case_id"] in cases: c=cases[r["case_id"]]; rows.append({**r,"oracle_action":c["oracle_action"],"task_success":task_ok(r,c)})
    for cond in sorted({r["condition"] for r in rows}):
        q=[r for r in rows if r["condition"]==cond]; ag,u,b,rr=summary(q); ts=np.mean([r["task_success"] for r in q]) if q else math.nan; fail=sum(r.get("action") not in ("reuse","refresh") for r in q); print(f"\n{cond}: n={len(q)} failed={fail} agreement={ag:.3f} task_success={ts:.3f} utility={u:.3f} bad_reuse={b:.3f} refresh={rr:.3f}")
        for reg in REGIMES:
            z=[r for r in q if r["regime"]==reg]
            if z: ag,u,b,rr=summary(z); print(f"  {reg:>10s}: agreement={ag:.3f} task_success={np.mean([r['task_success'] for r in z]):.3f} utility={u:.3f} bad_reuse={b:.3f}")
    by={}
    for r in rows: by.setdefault(r["case_id"],{})[r["condition"]]=r
    print("\npaired utility deltas")
    for left,right in (("raw","age_plane"),("raw","resolver"),("age_plane","resolver")):
        ds=[]; help=harm=0
        for cid,g in by.items():
            if left not in g or right not in g: continue
            x,y=g[left].get("action"),g[right].get("action")
            if x not in ("reuse","refresh") or y not in ("reuse","refresh"): continue
            o=cases[cid]["oracle_action"]; ds.append(reward(y,o)-reward(x,o)); help += int(x!=y and y==o); harm += int(x!=y and x==o)
        if ds: print(f"  {right}-{left}: mean={np.mean(ds):+.4f} helpful_flips={help} harmful_flips={harm} n={len(ds)}")

def parser():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
    x=s.add_parser("generate"); x.add_argument("--output",type=Path,required=True); x.add_argument("--per-regime",type=int,default=60); x.add_argument("--seed",type=int,default=42); x.set_defaults(f=cmd_generate)
    x=s.add_parser("sanity"); x.add_argument("--input",type=Path,required=True); x.set_defaults(f=cmd_sanity)
    x=s.add_parser("run-openai"); x.add_argument("--input",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.add_argument("--model",default="gpt-5"); x.add_argument("--conditions",nargs="+",default=list(CONDS),choices=CONDS); x.add_argument("--workers",type=int,default=6); x.add_argument("--limit",type=int); x.add_argument("--base-url"); x.set_defaults(f=cmd_run)
    x=s.add_parser("score"); x.add_argument("--cases",type=Path,required=True); x.add_argument("--responses",type=Path,required=True); x.set_defaults(f=cmd_score); return p
if __name__=="__main__":
    a=parser().parse_args(); a.f(a)
