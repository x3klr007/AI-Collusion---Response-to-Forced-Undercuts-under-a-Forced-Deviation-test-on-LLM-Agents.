"""Live-compatible paired Bertrand pricing experiment.

No synthetic results may be labelled empirical: meta.json explicitly records provider.
Supports a true before-cut fork where simultaneous model choices at period t are
identical across branches, and only the executed price is intervened upon.

Examples:
 python -m collusion.runner benchmark
 python -m collusion.runner base --provider mock --rounds 8 --runs 1
 python -m collusion.runner fork --provider mock --base-dir logs/base_... --period 4 --horizon 3
 python -m collusion.runner pretest --provider mock --rounds 50
 python -m collusion.runner scripted
Real API calls require BOTH a provider-specific environment key and --execute-paid.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import uuid
from pathlib import Path

from .client import get_client
from .market import Market
from .prompt import build_prompt, P1, GENERAL_PROMPT, SCHEMA
from .scripted import grid

ROOT = Path(__file__).resolve().parents[1]
LOGROOT = ROOT/"logs"
ACTORS=("Firm A","Firm B")
DEFAULT_MODEL={"anthropic":"claude-haiku-5-5","openai":"gpt-5-mini","mock":"synthetic-sticky-policy-v1"}

def digest(item) -> str:
    if isinstance(item,bytes): binary=item
    else: binary=json.dumps(item,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(binary).hexdigest()

def save_json(path: Path, obj) -> None:
    with path.open("x",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False)
        f.write("\n")

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        result=[json.loads(line) for line in f if line.strip()]
    if any(r["period"] != i+1 for i,r in enumerate(result)):
        raise ValueError("Noncontiguous or malformed history")
    return result

def append_round(path: Path, row: dict) -> None:
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
        f.flush()
        os.fsync(f.fileno())

def run_folder(kind: str) -> Path:
    LOGROOT.mkdir(parents=True,exist_ok=True)
    path=LOGROOT / f"{kind}_{uuid.uuid4().hex[:12]}"
    path.mkdir(exist_ok=False)
    return path

def market_data(name: str, prices: dict[str,float], quantities: dict, profits: dict) -> str:
    rivals={a:round(p,2) for a,p in prices.items() if a != name}
    chunks=[f"- My price: {round(prices[name],2)}"]
    if rivals:
        chunks.append(f"- Competitor's prices: {rivals}")
    chunks.extend((f"- My quantity sold: {round(quantities[name],2)}",
                   f"- My profit earned: {round(profits[name],2)}"))
    return "\n".join(chunks)+"\n"

def recalculate_round(record: dict, market: Market) -> None:
    prices={a:record["actors"][a]["executed_price"] for a in record["actors"]}
    qs,ps={},{}
    for a,p in prices.items():
        rival=next((rp for name,rp in prices.items() if name!=a),None)
        qs[a]=market.quantity(p,rival)
        ps[a]=market.profit(p,rival)
    for a in prices:
        record["actors"][a]["quantity"]=qs[a]
        record["actors"][a]["profit"]=ps[a]
        record["actors"][a]["market_data"]=market_data(a,prices,qs,ps)
    record["market_prices"]=prices
    record["total_profit"]=sum(ps.values())

async def one_round(history: list[dict], client, market: Market, period: int,
                    agents: tuple[str,...]) -> dict:
    prompts={a:build_prompt(history,a,marginal_cost=market.cost) for a in agents}
    answers=await asyncio.gather(*(client.choose(prompts[a],a) for a in agents))
    actors={}
    for a,(reply,meta) in zip(agents,answers):
        original=float(reply["chosen_price"])
        actors[a]={"prompt":prompts[a],"prompt_sha256":digest(prompts[a].encode()),
                   "response":reply,"model_chosen_price":original,
                   "executed_price":original,"model_meta":meta}
    record={"period":period,"actors":actors,"intervention":None}
    recalculate_round(record,market)
    return record

def common_meta(provider: str, model: str,kind: str, rounds: int, n_agents: int,seed: int):
    protocol=(ROOT/"PROTOCOL.md").read_bytes()
    return {"kind":kind,"provider":provider,"model":model,"seed_local_only":seed,
            "planned_rounds":rounds,"n_agents":n_agents,
            "market":Market().__dict__,
            "protocol_sha256":digest(protocol),
            "prompt_sha256":digest((P1+GENERAL_PROMPT+SCHEMA).encode()),
            "source_commit":"b210bfb0f8499de9d9710424d7570808e23a885a",
            "classification":"SYNTHETIC_SMOKE_ONLY" if provider=="mock" else "LIVE_MODEL",
            "no_model_seed_guarantee":provider!="mock"}

async def base_episode(client,provider: str,model: str,rounds: int,
                       *,mono: bool=False,seed: int=0) -> Path:
    agents=ACTORS[:1] if mono else ACTORS
    path=run_folder("pretest" if mono else "base")
    meta=common_meta(provider,model,"pretest" if mono else "base",rounds,len(agents),seed)
    save_json(path/"meta.json",meta)
    history=[]
    try:
        for t in range(1,rounds+1):
            r=await one_round(history,client,Market(),t,agents)
            history.append(r)
            append_round(path/"rounds.jsonl",r)
        manifest={"completed_rounds":len(history),
                  "rounds_sha256":digest((path/"rounds.jsonl").read_bytes()),
                  "final_prices":history[-1]["market_prices"],
                  "last10_mean":{a:sum(x["actors"][a]["executed_price"] for x in history[-10:])/min(10,len(history))
                                 for a in agents},
                  "classification":meta["classification"]}
        if mono:
            manifest["true_one_product_monopoly"]=Market().joint_monopoly(1)
            manifest["incorrect_two_product_monopoly_for_pretest"]=Market().joint_monopoly(2)
            manifest["pretest_near_one_product_monopoly"]=all(
                abs(p-manifest["true_one_product_monopoly"])<.12 for p in manifest["last10_mean"].values())
        save_json(path/"completion.json",manifest)
    except BaseException:
        # No completion marker for partial runs: preserve all primitive evidence.
        raise
    return path

def choose_fork_source(base: list[dict],period: int,agent: str,kind: str,market: Market) -> dict:
    if period<2 or period>len(base):
        raise ValueError("Fork period outside source run")
    original=copy.deepcopy(base[period-1])
    if agent not in original["actors"]:
        raise ValueError("Unknown cutter")
    original["source_round_sha256"]=digest(base[period-1])
    original["intervention"]={"kind":kind,"agent":agent,"period":period}
    if kind in ("cut","up"):
        prior_price=base[period-2]["actors"][agent]["executed_price"]
        changed=(.9 if kind=="cut" else 1.1)*prior_price
        original["actors"][agent]["executed_price"]=changed
        # Protocol says the cutter's own prompt history reports the forced price.
        original["actors"][agent]["response"]["chosen_price"]=changed
        original["intervention"].update({"original_model_price":base[period-1]["actors"][agent]["model_chosen_price"],
                                          "forced_price":changed,"previous_price":prior_price})
        recalculate_round(original,market)
    elif kind not in ("sham","sham2"):
        raise ValueError("Unknown branch kind")
    return original

async def fork_episode(client,provider: str,model: str,base_dir: Path,
                       period: int,horizon: int,cutter: str,kind: str,seed: int=0) -> Path:
    if horizon<1:raise ValueError("Horizon must be positive")
    bm=json.loads((base_dir/"meta.json").read_text())
    if bm["n_agents"]!=2:
        raise ValueError("Must fork a duopoly")
    if bm["model"]!=model or bm["provider"]!=provider:
        raise ValueError("Branch provider/model must match source")
    if bm["protocol_sha256"]!=digest((ROOT/"PROTOCOL.md").read_bytes()):
        raise ValueError("Source protocol digest differs; do not silently mix versions")
    base=load_jsonl(base_dir/"rounds.jsonl")
    if len(base)<period: raise ValueError("Missing base source checkpoint")
    prefix=copy.deepcopy(base[:period-1])
    source=choose_fork_source(base,period,cutter,kind,Market())
    path=run_folder("fork_"+kind)
    meta=common_meta(provider,model,"fork_"+kind,horizon,2,seed)
    meta.update({"base_path":str(base_dir.resolve()),"fork_period":period,"cutter":cutter,
                 "intervention":kind,"prefix_sha256":digest(base[:period-1]),
                 "source_round_sha256":digest(base[period-1]),
                 "branch_initial_prompt_equivalence":"identical for all branches"})
    save_json(path/"meta.json",meta)
    rounds=prefix+[source]
    append_round(path/"continuation.jsonl",source)
    for t in range(period+1,period+horizon+1):
        row=await one_round(rounds,client,Market(),t,ACTORS)
        rounds.append(row)
        append_round(path/"continuation.jsonl",row)
    comp={"complete":True,"period":period,"through_period":period+horizon,"kind":kind,
          "prefix_sha256":meta["prefix_sha256"],
          "records_sha256":digest((path/"continuation.jsonl").read_bytes())}
    save_json(path/"completion.json",comp)
    return path

def script_grid() -> None:
    rows=grid(repeats=50,noise_sd=.01)
    print(f"SYNTHETIC GRID: {len(rows)} cells, {sum(x['classified']==x['expected'] for x in rows)}/24 pass")
    p=run_folder("offline_scripted")
    with (p/"grid.csv").open("x",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(p/"completion.json",{"classification":"SYNTHETIC_CONTROL_NOT_LLM",
              "cells":len(rows),"repetitions":50,"pass":sum(x['classified']==x['expected'] for x in rows),
              "grid_sha256":digest((p/"grid.csv").read_bytes())})
    print("Evidence:",p)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("benchmark","scripted","base","pretest","fork"))
    parser.add_argument("--provider",choices=("mock","anthropic","openai"),default="mock")
    parser.add_argument("--model",default=None)
    parser.add_argument("--rounds",type=int,default=150)
    parser.add_argument("--runs",type=int,default=1)
    parser.add_argument("--seed",type=int,default=20261025)
    parser.add_argument("--execute-paid",action="store_true")
    parser.add_argument("--base-dir",type=Path)
    parser.add_argument("--pretest-dir",type=Path)
    parser.add_argument("--period",type=int,default=75)
    parser.add_argument("--horizon",type=int,default=15)
    parser.add_argument("--cutter",choices=ACTORS,default="Firm A")
    parser.add_argument("--kind",choices=("cut","sham","sham2","up"),default="cut")
    args=parser.parse_args()
    if args.command=="benchmark":
        print(json.dumps(Market(quantity_scale=1).reference(),indent=2))
        return
    if args.command=="scripted":
        script_grid()
        return
    if args.provider!="mock" and not args.execute_paid:
        parser.error("Real model calls are disabled by default: --execute-paid required")
    if args.command=="fork" and not args.base_dir:
        parser.error("--base-dir required for fork")
    model=args.model or DEFAULT_MODEL[args.provider]
    if args.command=="base" and args.provider!="mock":
        if args.pretest_dir is None:
            parser.error("Live base runs require --pretest-dir with a qualified 50-period solo trial")
        pre_meta=json.loads((args.pretest_dir/"meta.json").read_text())
        pre_done=json.loads((args.pretest_dir/"completion.json").read_text())
        if (pre_meta["kind"]!="pretest" or pre_meta["provider"]!=args.provider
             or pre_meta["model"]!=model or pre_meta["planned_rounds"]<50
             or pre_meta["protocol_sha256"]!=digest((ROOT/"PROTOCOL.md").read_bytes())
             or not pre_done.get("pretest_near_one_product_monopoly",False)):
            parser.error("Pretest does not qualify this model, provider and protocol")
    async def do_run():
        paths=[]
        for i in range(args.runs if args.command in ("base","pretest") else 1):
            client=get_client(args.provider,model,seed=args.seed+i)
            try:
                if args.command=="fork":
                    p=await fork_episode(client,args.provider,model,args.base_dir,args.period,
                                         args.horizon,args.cutter,args.kind,args.seed)
                else:
                    p=await base_episode(client,args.provider,model,
                                         args.rounds,mono=(args.command=="pretest"),seed=args.seed+i)
                paths.append(str(p))
            finally:
                if hasattr(client,"close"): await client.close()
        for p in paths: print("COMPLETED:",p)
    asyncio.run(do_run())

if __name__=="__main__":
    main()
