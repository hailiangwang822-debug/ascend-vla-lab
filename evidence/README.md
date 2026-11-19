# Validation and experiment records

## StarVLA training integration

`private-prototype-validation.json` records local tests on the previous prototype, including the full Spatial CPU reader (432 episodes / 52,970 frames), tokenizer checks, and seven recovery tests under installed Accelerate 1.16.0 and target Accelerate 1.11.0 loaded from its official source tag. It does not record a real NPU training run.

The public version changes OBS paths into a required `STARVLA_OBS_ROOT` setting and adds a root-validation test. See `public-validation.json` for its actual local execution result. The small CPU test compares uninterrupted training against save-at-7/resume-to-20 using dropout, Adam, a scheduler and data cursor. It is a mechanism test; it does not establish bitwise equivalence on distributed NPUs.

Pending: real ModelArts OBS integration, HCCL startup, NPU operators, actual memory/time, 64-card recovery and trained policy evaluation.

## Separate π0.5 inference work

The user supplied a fixed-task evaluation summary of 27 LIBERO Spatial episodes: 24 successes, 3 timeouts, 0 errors. This is a custom initial-state subset, not the full LIBERO benchmark or a reproduction of an official paper score.

After reducing executed action steps from 50 to 10, the user reported six successes on a targeted six-episode diagnostic. The protocol is inferred from the active experiment instructions; raw records have not been received or independently verified. Do not combine it with the 27-episode result or describe it as a StarVLA training improvement.

## Evidence to collect on the cloud

Preserve the exact code/release, asset hashes, environment versions, launch configuration, step/loss/throughput logs, committed checkpoint markers, restore log, final model inventory, evaluation plan, per-episode JSON and failure/success videos. Document incomplete runs as incomplete.
