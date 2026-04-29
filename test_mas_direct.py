#!/usr/bin/env python3
"""Direct MAS pipeline test — bypasses dataset loading, exercises full latent communication."""
import os
import sys
import argparse
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from inference_utils.inference_mas import (
    resolve_dtype,
    run_planner_latent_stage,
    run_refiner_latent_stage,
    run_solver_latent_stage,
    format_latent_info,
    release_resources,
)

# ── Paths ──────────────────────────────────────────────────────────
SNAP = "/root/.cache/huggingface/hub"
PLANNER = os.path.join(SNAP, "models--RecursiveMAS--Sequential-Light-Planner-Qwen3-1.7B/snapshots/98ba7ec8230e1318ac4de9bfb5d5e85f341b0307")
CRITIC  = os.path.join(SNAP, "models--RecursiveMAS--Sequential-Light-Critic-Llama3.2-1B/snapshots/b24d06be9de449803f23f712c86646d27444036c")
SOLVER  = os.path.join(SNAP, "models--RecursiveMAS--Sequential-Light-Solver-Qwen2.5-Math-1.5B/snapshots/fa89c80170896e5be0da362fb2d87153b7d58cbc")
OUTER   = os.path.join(SNAP, "models--RecursiveMAS--Sequential-Light-Outerlinks/snapshots/12420b91249efe1d05cf80b72de7d8007aa85b00")

OUTER_12 = os.path.join(OUTER, "Planner-Critic-Outerlink(math).pt")
OUTER_23 = os.path.join(OUTER, "Critic-Solver-Outerlink(math).pt")
OUTER_31 = os.path.join(OUTER, "Solver-Planner-Outerlink(math).pt")
INNER_P  = os.path.join(PLANNER, "adapter(math).pt")
INNER_C  = os.path.join(CRITIC, "adapter(math).pt")
INNER_S  = os.path.join(SOLVER, "adapter(math).pt")

# ── Test questions ──────────────────────────────────────────────────
QUESTIONS = [
    "Find the sum of all integers n such that n^2 - 3n + 2 is a prime number.",
    "If x + y = 10 and xy = 21, find the value of x^2 + y^2.",
]

# ── Dummy args for solver stage ─────────────────────────────────────
class DummyArgs:
    mas_shape = "chain"
    solver_pre_question = 0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype("auto") or torch.bfloat16
    outer_dtype = resolve_dtype("auto") or torch.float32

    trust_remote_code = True
    enable_thinking = False
    latent_steps = 4
    batch_size = 1
    max_new_tokens = 512
    inner_fb = "ln_res_adapter"
    outer_fb = "outer_ln_res_adapter"
    args = DummyArgs()

    print(f"[test] device={device}, model_dtype={model_dtype}, outer_dtype={outer_dtype}")
    print(f"[test] questions: {len(QUESTIONS)}")

    # Stage 1: Planner → latent → outer_12 → Refiner input
    print("\n── Stage 1: Planner latent ──")
    p2r = run_planner_latent_stage(
        model_name_or_path=PLANNER, questions=QUESTIONS,
        agent1_inner_aligner_path=INNER_P,
        outer_12_path=OUTER_12, outer_12_type=outer_fb,
        latent_steps=latent_steps, batch_size=batch_size,
        device=device, model_dtype=model_dtype, outer_dtype=outer_dtype,
        trust_remote_code=trust_remote_code,
        inner_adapter_type_fallback=inner_fb,
        enable_thinking=enable_thinking,
    )
    for i, t in enumerate(p2r):
        print(f"  p2r[{i}]: shape={tuple(t.shape)}, norm={t.norm().item():.4f}")

    # Stage 2: Refiner receives planner latent → latent rollout → outer_23 → Solver input
    print("\n── Stage 2: Refiner latent ──")
    r2s = run_refiner_latent_stage(
        model_name_or_path=CRITIC, questions=QUESTIONS,
        planner_latents=p2r, agent2_inner_aligner_path=INNER_C,
        outer_23_path=OUTER_23, outer_23_type=outer_fb,
        latent_steps=latent_steps, batch_size=batch_size,
        device=device, model_dtype=model_dtype, outer_dtype=outer_dtype,
        trust_remote_code=trust_remote_code,
        inner_adapter_type_fallback=inner_fb,
        enable_thinking=enable_thinking,
    )
    for i, t in enumerate(r2s):
        print(f"  r2s[{i}]: shape={tuple(t.shape)}, norm={t.norm().item():.4f}")

    # Stage 3: Solver receives refiner latent → generates TEXT answer
    print("\n── Stage 3: Solver generation ──")
    answers = run_solver_latent_stage(
        model_name_or_path=SOLVER, questions=QUESTIONS,
        refiner_latents=r2s, args=args,
        batch_size=batch_size, max_new_tokens=max_new_tokens,
        do_sample=True, temperature=0.6, top_p=0.95,
        device=device, dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        enable_thinking=enable_thinking,
    )
    for i, ans in enumerate(answers):
        print(f"  [{i}] Q: {QUESTIONS[i]}")
        print(f"      A: {ans[:300]}...")

    print("\n── PASS: Full MAS pipeline completed ──")

if __name__ == "__main__":
    main()
