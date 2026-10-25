"""Haiku 5.5 and cost preflight tests; HTTP mock transport only; zero paid calls."""
from __future__ import annotations

import asyncio
import httpx
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from collusion.runner import DEFAULT_MODEL
from collusion.client import APIClient
from collusion.preflight_budget import (
    stage_counts,estimate_campaign,scan_usage_records,estimate_cost
)

class Haiku55Contract(unittest.TestCase):
    def test_latest_default_matches_amendment_a7(self):
        self.assertEqual(DEFAULT_MODEL["anthropic"],"claude-haiku-5-5")

    def test_valid_json_after_thinking_block_on_haiku55(self):
        observed=[]
        async def handler(req:httpx.Request):
            sent=json.loads(req.content)
            observed.append(sent)
            self.assertEqual(sent["model"],"claude-haiku-5-5")
            self.assertNotIn("temperature",sent)
            self.assertNotIn("top_p",sent)
            self.assertNotIn("top_k",sent)
            self.assertEqual(sent["output_config"],{"effort":"medium"})
            self.assertEqual(sent["max_tokens"],1200)
            return httpx.Response(200,json={
                "model":"claude-haiku-5-5",
                "id":"msg_fake_01",
                "type":"message",
                "content":[
                  {"type":"thinking","thinking":"synthetic not real"},
                  {"type":"text","text":json.dumps({
                     "chosen_price":1.8,"plans":"hold", "insights":"hold",
                     "observations":"synthetic"}
                   )}],
                "stop_reason":"end_turn",
                "usage":{"input_tokens":3500,"output_tokens":620}
            })
        async def run():
            with patch.dict("os.environ",{"ANTHROPIC_API_KEY":"invalid-test-only"}):
                client=APIClient("anthropic",DEFAULT_MODEL["anthropic"])
            await client.http.aclose()
            client.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                reply,meta=await client.choose("synthetic price context","Firm B")
            finally:
                await client.close()
            self.assertEqual(reply["chosen_price"],1.8)
            self.assertEqual(meta["resolved_model"],"claude-haiku-5-5")
            self.assertEqual(meta["stop_reason"],"end_turn")
            self.assertEqual(meta["provider_response_id"],"msg_fake_01")
            self.assertEqual(meta["usage"]["output_tokens"],620)
        asyncio.run(run())
        self.assertEqual(len(observed),1)

    def test_original_4_5_temperature_behavior_retained(self):
        async def handler(req:httpx.Request):
            sent=json.loads(req.content)
            self.assertEqual(sent["temperature"],.7)
            self.assertNotIn("output_config",sent)
            return httpx.Response(200,json={
                "content":[{"type":"text","text":json.dumps({
                    "chosen_price":1.70,"plans":"","insights":""})}],
                "usage":{"input_tokens":10,"output_tokens":30}
            })
        async def run():
            with patch.dict("os.environ",{"ANTHROPIC_API_KEY":"invalid-test-only"}):
                client=APIClient("anthropic","claude-haiku-4-5-20251001")
            await client.http.aclose()
            client.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                answer,meta=await client.choose("toy fixture","Firm A")
            finally:
                await client.close()
            self.assertEqual(answer["chosen_price"],1.7)
        asyncio.run(run())

class BudgetTests(unittest.TestCase):
    def test_amended_call_counts_and_budget(self):
        stages=stage_counts()
        self.assertEqual(stages,{
          "solo_pretest":50,"two_agent_baselines":3600,
          "cut_and_sham_continuations":2880,
          "second_sham_controls":360,
          "upward_controls":0
        })
        self.assertEqual(sum(stages.values()),6890)
        self.assertEqual(sum(stage_counts(include_up=True).values()),7250)
        report=estimate_campaign(credits_usd=3.0,
                                 input_tokens=3500,output_tokens=600,
                                 reserve_fraction=.15)
        self.assertAlmostEqual(report["expected_spend_usd"],4.4785)
        self.assertAlmostEqual(report["available_for_spend_after_reserve_usd"],2.55)
        self.assertFalse(report["full_plan_fits_assumed_budget"])
        self.assertAlmostEqual(report["estimated_additional_budget_required_usd"],1.9285)
        self.assertTrue(estimate_campaign(credits_usd=6)["full_plan_fits_assumed_budget"])

    def test_budget_estimates_long_prompt_rate_tier(self):
        self.assertAlmostEqual(estimate_cost(1,3500,600),.00065)
        self.assertAlmostEqual(estimate_cost(1,101000,600),
                               (101000*.5+600*2.5)/1e6)
        self.assertRaises(ValueError,estimate_cost,1,-1,600)
        self.assertRaises(ValueError,estimate_campaign,
                          credits_usd=-1)

    def test_usage_file_read_only_and_mock_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"usage.jsonl"
            actors=[{"model_meta":{"provider":"anthropic",
                        "usage":{"input_tokens":2900,"output_tokens":350}}},
                    {"model_meta":{"provider":"anthropic",
                        "usage":{"input_tokens":4100,"output_tokens":700}}}]
            record={"actors":{str(i):actor for i,actor in enumerate(actors)}}
            path.write_text(json.dumps(record)+"\n")
            sample=scan_usage_records(path)
            self.assertEqual(sample["n_observed_paid_replies"],2)
            self.assertEqual(sample["mean_input_tokens"],3500)
            self.assertEqual(sample["mean_output_tokens"],525)
            res=estimate_campaign(credits_usd=3,observed=sample)
            self.assertEqual(res["usage_source"],"live_usage_sample")
            self.assertGreater(res["expected_spend_usd"],0)

class FailFastTests(unittest.TestCase):
    def test_provider_bad_request_is_not_retried(self):
        counter=[0]
        async def handler(req):
            counter[0]+=1
            return httpx.Response(400,json={"type":"error","error":{"message":"invalid request"}})
        async def run():
            with patch.dict("os.environ",{"ANTHROPIC_API_KEY":"invalid-test-only"}):
                client=APIClient("anthropic","claude-haiku-5-5")
            await client.http.aclose()
            client.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                with self.assertRaisesRegex(RuntimeError,"HTTP 400"):
                    await client.choose("fake","Firm B")
            finally:
                await client.close()
        asyncio.run(run())
        self.assertEqual(counter[0],1)

    def test_exhausted_thinking_budget_does_not_retry_or_accept_partial(self):
        counter=[0]
        async def handler(req):
            counter[0]+=1
            return httpx.Response(200,json={
                "model":"claude-haiku-5-5",
                "content":[{"type":"thinking","thinking":"placeholder"}],
                "stop_reason":"max_tokens",
                "usage":{"input_tokens":10,"output_tokens":1200}
            })
        async def run():
            with patch.dict("os.environ",{"ANTHROPIC_API_KEY":"invalid-test-only"}):
                client=APIClient("anthropic","claude-haiku-5-5")
            await client.http.aclose()
            client.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                with self.assertRaisesRegex(RuntimeError,"max_tokens"):
                    await client.choose("fake","Firm B")
            finally:
                await client.close()
        asyncio.run(run())
        self.assertEqual(counter[0],1)

if __name__=="__main__":
    unittest.main()
