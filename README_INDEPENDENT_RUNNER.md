# Independent runner — execution instructions

This local development branch adds a standalone Python runner, 12-run campaign driver, offline scripted controls, and inference/tests under collusion/ and tests/. Original team PROTOCOL.md, PREDICTIONS.md, BRIEF.md and part1/scripted_sim.py are unchanged.

All commands run inside the working-copy directory.

## Offline, zero cost

    python3 -m unittest discover -s tests -v
    python3 -m collusion.runner benchmark
    python3 -m collusion.runner scripted
    python3 -m collusion.runner base --provider mock --rounds 8 --runs 1

To rehearse a paired fork, use the printed base directory:

    python3 -m collusion.runner fork --provider mock --base-dir logs/base_ID --period 4 --horizon 3 --kind sham
    python3 -m collusion.runner fork --provider mock --base-dir logs/base_ID --period 4 --horizon 3 --kind cut

To validate and recompute the paired prices and profits:

    python3 -m collusion.analysis --base-dir logs/base_ID --sham-dir logs/fork_sham_ID --cut-dir logs/fork_cut_ID --rho 0.85

A complete 12x150 mock census exists in logs/full_mock_census.txt. Its 48-point synthetic fork campaign and extra controls are stored under logs/campaign_8eca4197de97/. Use the corrected descriptive summary, not the preserved earlier invalid regression.

## Amendment A7 — budget and Haiku 5.5 compatibility

Before touching paid API credentials, execute:

    python3 -m collusion.preflight_budget --credits-usd 3 --strict-budget

This zero-network preflight uses adjustable assumptions of 3,500 input tokens and 600 output tokens per call, and a 15% reserve. The full amended plan has 6,890 pricing-agent decisions and is not expected to fit $3 of credits under these assumptions. It is a planning aid, **not** a provider-enforced spending cap. After a bounded live pilot, update the token assumptions from actual usage before authorizing the full paid design.

The current main model is Claude Haiku 5.5 as required by Amendment A7. Its request omits temperature, top_p and top_k, freezes medium effort, and records the provider-reported model ID and token usage. With adaptive thinking, max_tokens=1200 may stop before JSON price output; such requests are rejected rather than silently accepted. Provider compatibility has only been tested with a fake HTTP transport, not paid inference.

## Live model — NOT YET EXECUTED

Prerequisites: sign off the scientific corrections in INDEPENDENT_EXECUTION_HANDOFF_20261025.md; establish a monetary ceiling and API permission; configure the ANTHROPIC_API_KEY environment variable locally on the workstation without sharing it in chat. Every real model command explicitly requires --execute-paid.

Step 1: a correct 50-period one-product solo pretest:

    python3 -m collusion.runner pretest --provider anthropic --model claude-haiku-5-5 --rounds 50 --execute-paid

Read its completion.json. It must have pretest_near_one_product_monopoly=true.

Step 2: twelve independent 150-period duopoly baselines (qualified pretest path is mandatory):

    python3 -m collusion.runner base --provider anthropic --model claude-haiku-5-5 --rounds 150 --runs 12 --pretest-dir logs/pretest_ID --execute-paid | tee logs/live_base_census.txt

Step 3: freeze and execute the matched interventions. Provider/model/pretest must agree and live baselines must end above Nash.

    python3 -m collusion.campaign --base-list logs/live_base_census.txt --provider anthropic --model claude-haiku-5-5 --execute-paid

Step 4: primitive log recomputation, cluster-bootstrap rho re-estimation, and abstention checks.

    python3 -m collusion.inference --campaign logs/campaign_ID --resamples 10000

The agent's original simultaneous period-t choices are reused in both branches; only the specified executed price and subsequent market history are modified. All generated logs reside in ignored logs/ and include explicit provider/synthetic labels.

Scientific scope: negative PI_dyn identifies a deviation from a particular sticky-myopic best-response null. It is not proof of intentional punishment. The forecasting-based out-of-model test illustrates this limitation.
