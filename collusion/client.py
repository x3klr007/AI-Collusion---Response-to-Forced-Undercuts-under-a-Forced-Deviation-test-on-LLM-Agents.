"""Experiment pricing clients. No API call without --provider anthropic/openai.

Only ANTHROPIC_API_KEY / OPENAI_API_KEY are read at runtime, never logged.
The mock provider is for pipeline integrity testing and is NOT LLM evidence.
"""
from __future__ import annotations
import asyncio
import hashlib
import json
import math
import os
import random
import re
from typing import Any

from .market import Market


class NonRetryableProviderResponse(RuntimeError):
    """A provider error that repeating unchanged cannot repair or qualify."""


def validate_reply(raw: str) -> dict:
    try:
        item = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Not a single valid JSON object: {exc}") from exc
    if not isinstance(item,dict) or isinstance(item.get("chosen_price"),bool):
        raise ValueError("Expected object containing numeric chosen_price")
    try: price = float(item["chosen_price"])
    except (TypeError,ValueError,KeyError) as exc:
        raise ValueError("Invalid price") from exc
    if not (math.isfinite(price) and 0 < price < 20):
        raise ValueError(f"Out-of-range price: {price}")
    return {
        "chosen_price": price,
        "plans": str(item.get("plans","")),
        "insights": str(item.get("insights","")),
        "observations": str(item.get("observations",""))
    }

class MockClient:
    provider = "mock"
    model = "synthetic-sticky-policy-v1"
    def __init__(self, *, seed: int=0) -> None:
        self.seed = seed
        self.market = Market()
    async def choose(self, prompt: str, agent: str) -> tuple[dict,dict]:
        # Minimal deterministic local stand-in: response driven by market history
        # (different prompt => different seed), but NO real language-model inference.
        # Use regex on the latest source-equivalent market-data block.
        own = re.search(r"- My price: ([0-9.]+)",prompt)
        other = re.search(r"- Competitor's prices: \{[^:]+: ([0-9.]+)\}",prompt)
        prev = float(own.group(1)) if own else 1.9
        rival = float(other.group(1)) if other else 1.9
        target = self.market.best_response(rival) + (1.9-self.market.best_response(1.9))
        base = .85*prev + .15*target
        # Stable per (prompt,agent,seed), suitable for deterministic integrity tests.
        digest=hashlib.sha256(f"{self.seed}|{agent}|{prompt}".encode()).digest()
        noise=(int.from_bytes(digest[:4],"big")/2**32-.5)*.006
        value=max(.50,min(4.51,base+noise))
        response={"chosen_price":round(value,4),
                  "plans":"Synthetic fixture: adjust price conservatively.",
                  "insights":"Synthetic test only; no language model was called.",
                  "observations":""}
        return response,{"provider":self.provider,"model":self.model,"usage":{}}

class APIClient:
    def __init__(self,provider: str,model: str,temperature: float=.7,
                 max_tokens: int=1200) -> None:
        if provider not in ("anthropic","openai"):
            raise ValueError("API provider must be anthropic or openai")
        key_name="ANTHROPIC_API_KEY" if provider=="anthropic" else "OPENAI_API_KEY"
        self.key=os.environ.get(key_name,"")
        if not self.key:
            raise RuntimeError(f"Missing {key_name}; no paid calls attempted")
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens = temperature, max_tokens
        import httpx
        self.http=httpx.AsyncClient(timeout=120)
    async def close(self) -> None:
        await self.http.aclose()
    async def choose(self,prompt: str,agent: str) -> tuple[dict,dict]:
        from .prompt import P1,SCHEMA
        if self.provider=="anthropic":
            url="https://api.anthropic.com/v1/messages"
            headers={"x-api-key":self.key,"anthropic-version":"2023-06-01"}
            payload={"model":self.model,"max_tokens":self.max_tokens,
                     "system":SCHEMA+"\n"+P1,
                     "messages":[{"role":"user","content":prompt}]}
            if self.model.startswith("claude-haiku-5-5"):
                # Haiku 5.5 rejects sampling-parameter overrides (temperature,
                # top_p and top_k). Explicitly freeze its documented default
                # medium effort; thinking tokens share the max_tokens budget.
                payload["output_config"]={"effort":"medium"}
            else:
                # Preserve the older 4.5 protocol if chosen explicitly.
                payload["temperature"]=self.temperature
        else:
            url="https://api.openai.com/v1/chat/completions"
            headers={"Authorization":"Bearer "+self.key}
            payload={"model":self.model,"max_completion_tokens":self.max_tokens,
                     "messages":[{"role":"system","content":SCHEMA+"\n"+P1},
                                 {"role":"user","content":prompt}],
                     "response_format":{"type":"json_object"}}
        # Do not log payload headers/credential. Only safe error summaries.
        for attempt in range(3):
            try:
                resp=await self.http.post(url,headers=headers,json=payload)
                if resp.status_code in (408,429,500,502,503,504,529) and attempt<2:
                    await asyncio.sleep(2**attempt)
                    continue
                if 400 <= resp.status_code < 500 and resp.status_code not in (408,429):
                    # Bad parameters, nonexistent models and bad credentials
                    # must not consume three retries or disguise their cause.
                    raise NonRetryableProviderResponse(
                        f"HTTP {resp.status_code}; check model, credentials or request parameters"
                    )
                resp.raise_for_status()
                data=resp.json()
                if self.provider=="anthropic" and data.get("stop_reason") in (
                    "max_tokens","model_context_window_exceeded"):
                    # Haiku 5.5 medium-effort thinking shares max_tokens.
                    # Do not accept a clipped JSON decision or retry unchanged.
                    raise NonRetryableProviderResponse(
                        f"response ended with {data['stop_reason']}; raise max_tokens or reduce effort in a documented amendment"
                    )
                if self.provider=="anthropic":
                    blocks=data.get("content",[])
                    content="\n".join(b.get("text","") for b in blocks if b.get("type")=="text")
                    usage=data.get("usage",{})
                else:
                    content=data["choices"][0]["message"]["content"]
                    usage=data.get("usage",{})
                return validate_reply(content),{
                    "provider":self.provider,
                    "model":self.model,
                    "resolved_model":data.get("model",self.model),
                    "provider_response_id":data.get("id"),
                    "stop_reason":data.get("stop_reason"),
                    "usage":usage,
                    "retry_attempts":attempt
                }
            except NonRetryableProviderResponse as exc:
                raise RuntimeError(
                    f"Non-retryable model response; provider={self.provider}: {exc}"
                ) from exc
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Model request failed; provider={self.provider},"
                                       f" exception={type(exc).__name__}. No raw key or response logged.") from exc
                await asyncio.sleep(2**attempt)
        raise RuntimeError("Unreachable")

def get_client(provider: str,model: str, *, seed: int=0):
    if provider=="mock": return MockClient(seed=seed)
    return APIClient(provider,model)
