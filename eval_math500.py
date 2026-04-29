#!/usr/bin/env python3
"""Full MAS evaluation on MATH-500 using sequential_light latent communication."""
import os
import sys
import json
import argparse
import time
import torch
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from inference_utils.inference_mas import (
    resolve_dtype,
    run_planner_latent_stage,
    run_refiner_latent_stage,
    run_solver_latent_stage,
    run_answer_retry_stage,
    release_resources,
)
from inference_utils.answer_utils import extract_boxed_answer, compare_answers

# ── Paths ──────────────────────────────────────────────────────────
SNAP = "/root/.cache/huggingface/hub"
MODEL_PLANNER = os.path.join(SNAP, "models--RecursiveMAS--Sequential-Light-Planner-Qwen3-1.7B/snapshots/98ba7ec8230e1318ac4de9bfb5d5e85f341b0307")
MODEL_CRITIC  = os.path.join(SNAP, "models--RecursiveMAS--Sequential-Light-Critic-Llama3.2-1B/snapshots/b24d06be9de449803f23f712c86646d27444036c")
MODEL_SOLVER  = os.path.join(SNAP, "models--RecursiveMAS--Sequential-Light-Solver-Qwen2.5-Math-1.5B/snapshots/fa89c80170896e5be0da362fb2d87153b7d58cbc")
MODEL_OUTER   = os.path.join(SNAP, "models--RecursiveMAS--Sequential-Light-Outerlinks/snapshots/12420b91249efe1d05cf80b72de7d8007aa85b00")

OUTER_12 = os.path.join(MODEL_OUTER, "Planner-Critic-Outerlink(math).pt")
OUTER_23 = os.path.join(MODEL_OUTER, "Critic-Solver-Outerlink(math).pt")
OUTER_31 = os.path.join(MODEL_OUTER, "Solver-Planner-Outerlink(math).pt")
INNER_P  = os.path.join(MODEL_PLANNER, "adapter(math).pt")
INNER_C  = os.path.join(MODEL_CRITIC, "adapter(math).pt")
INNER_S  = os.path.join(MODEL_SOLVER, "adapter(math).pt")

DATA_PATH = os.path.join(SNAP, "datasets--HuggingFaceH4--MATH-500/snapshots")
DATA_FILE = None
if os.path.isdir(DATA_PATH):
    dirs = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    if dirs:
        DATA_FILE = os.path.join(DATA_PATH, dirs[0], "test.jsonl")
if DATA_FILE is None or not os.path.isfile(DATA_FILE):
    import glob
    candidates = glob.glob(os.path.join(SNAP, "datasets--HuggingFaceH4--MATH-500/snapshots/*/test.jsonl"))
    DATA_FILE = candidates[0] if candidates else None

DATASET_NAME = "huggingfaceh4/math-500"

# Dummy args for solver
class DummyArgs:
    mas_shape = "chain"
    solver_pre_question = 0


def load_math500(jsonl_path: str, max_samples: int = -1):
    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            problems.append((row["problem"], row["answer"], row["solution"]))
            if max_samples > 0 and len(problems) >= max_samples:
                break
    return problems


def process_batch(
    questions,
    gold_answers,
    device, model_dtype, outer_dtype,
    latent_steps, batch_size, max_new_tokens,
    temperature, top_p,
):
    inner_fb = "ln_res_adapter"
    outer_fb = "outer_ln_res_adapter"
    trust_rc = True
    et = False  # enable_thinking
    args = DummyArgs()

    # Stage 1: Planner latent → outer_12
    p2r = run_planner_latent_stage(
        model_name_or_path=MODEL_PLANNER,
        questions=questions,
        agent1_inner_aligner_path=INNER_P,
        outer_12_path=OUTER_12,
        outer_12_type=outer_fb,
        latent_steps=latent_steps,
        batch_size=batch_size,
        device=device,
        model_dtype=model_dtype,
        outer_dtype=outer_dtype,
        trust_remote_code=trust_rc,
        inner_adapter_type_fallback=inner_fb,
        enable_thinking=et,
    )

    # Stage 2: Refiner latent → outer_23
    r2s = run_refiner_latent_stage(
        model_name_or_path=MODEL_CRITIC,
        questions=questions,
        planner_latents=p2r,
        agent2_inner_aligner_path=INNER_C,
        outer_23_path=OUTER_23,
        outer_23_type=outer_fb,
        latent_steps=latent_steps,
        batch_size=batch_size,
        device=device,
        model_dtype=model_dtype,
        outer_dtype=outer_dtype,
        trust_remote_code=trust_rc,
        inner_adapter_type_fallback=inner_fb,
        enable_thinking=et,
    )

    # Stage 3: Solver generates text
    outputs = run_solver_latent_stage(
        model_name_or_path=MODEL_SOLVER,
        questions=questions,
        refiner_latents=r2s,
        args=args,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_rc,
        enable_thinking=et,
    )

    # Answer retry for missing boxed answers
    outputs, retry_count = run_answer_retry_stage(
        model_name_or_path=MODEL_SOLVER,
        outputs=outputs,
        dataset_name=DATASET_NAME,
        batch_size=batch_size,
        device=device,
        dtype=model_dtype,
        trust_remote_code=trust_rc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=16,
    )

    return outputs, retry_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--latent_steps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    if DATA_FILE is None:
        print("ERROR: Cannot find MATH-500 data file.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = resolve_dtype("auto") or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    outer_dtype = resolve_dtype("auto") or torch.float32

    print(f"Device: {device} | model_dtype: {model_dtype} | outer_dtype: {outer_dtype}")
    print(f"Latent steps: {args.latent_steps} | Batch: {args.batch_size} | Max new tokens: {args.max_new_tokens}")

    problems = load_math500(DATA_FILE, max_samples=args.num_samples)
    questions     = [p[0] for p in problems]
    gold_answers  = [p[1] for p in problems]
    print(f"Loaded {len(problems)} problems from MATH-500")

    correct = 0
    total = 0
    results = []

    total_samples = len(questions)
    for start in tqdm(range(0, total_samples, args.batch_size), desc="MAS eval"):
        end = min(start + args.batch_size, total_samples)
        batch_q = questions[start:end]
        batch_g = gold_answers[start:end]

        t0 = time.time()
        try:
            outputs, retries = process_batch(
                questions=batch_q,
                gold_answers=batch_g,
                device=device,
                model_dtype=model_dtype,
                outer_dtype=outer_dtype,
                latent_steps=args.latent_steps,
                batch_size=len(batch_q),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            print(f"\n[ERROR] batch {start}-{end}: {e}")
            for i in range(start, end):
                results.append({"idx": i, "correct": False, "error": str(e)})
            total += len(batch_q)
            continue

        elapsed = time.time() - t0

        for local_i in range(len(batch_q)):
            global_i = start + local_i
            pred_text = outputs[local_i]
            gold = batch_g[local_i]

            _, _, is_correct, _, _ = compare_answers(gold, pred_text, DATASET_NAME)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "idx": global_i,
                "question": batch_q[local_i][:120],
                "gold": gold,
                "pred_boxed": extract_boxed_answer(pred_text),
                "correct": is_correct,
            })

        # Progress
        acc = 100.0 * correct / total if total > 0 else 0
        tqdm.write(f"  batch {start}-{end}: {elapsed:.1f}s | acc={correct}/{total}={acc:.1f}%")

    final_acc = 100.0 * correct / total if total > 0 else 0
    print(f"\n{'='*50}")
    print(f"FINAL: {correct}/{total} = {final_acc:.2f}%")
    print(f"{'='*50}")

    # Save detailed results
    out_path = os.path.join(THIS_DIR, "math500_results.json")
    with open(out_path, "w") as f:
        json.dump({"accuracy": final_acc, "correct": correct, "total": total, "details": results}, f, indent=2)
    print(f"Results saved to: {out_path}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
