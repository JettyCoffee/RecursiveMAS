#!/usr/bin/env python3
"""Quick test script to verify RecursiveMAS sequential_light pipeline."""
import os
import sys

# Env setup
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MAS_FORCE_DISABLE_TORCHVISION", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Ensure RecursiveMAS is importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

# Model snapshot paths
SNAPSHOT_BASE = "/root/.cache/huggingface/hub"
PLANNER = os.path.join(SNAPSHOT_BASE, "models--RecursiveMAS--Sequential-Light-Planner-Qwen3-1.7B/snapshots/98ba7ec8230e1318ac4de9bfb5d5e85f341b0307")
CRITIC  = os.path.join(SNAPSHOT_BASE, "models--RecursiveMAS--Sequential-Light-Critic-Llama3.2-1B/snapshots/b24d06be9de449803f23f712c86646d27444036c")
SOLVER  = os.path.join(SNAPSHOT_BASE, "models--RecursiveMAS--Sequential-Light-Solver-Qwen2.5-Math-1.5B/snapshots/fa89c80170896e5be0da362fb2d87153b7d58cbc")
OUTER   = os.path.join(SNAPSHOT_BASE, "models--RecursiveMAS--Sequential-Light-Outerlinks/snapshots/12420b91249efe1d05cf80b72de7d8007aa85b00")

sys.argv = [
    "inference_mas.py",
    "--mas_shape", "chain",
    "--dataset", "HuggingFaceH4/MATH-500",
    "--dataset_split", "test",
    "--num_samples", "5",
    "--seed", "42",
    "--num_recursive_rounds", "1",
    "--batch_size", "1",
    "--latent_steps", "4",
    "--max_new_tokens", "1024",
    "--temperature", "0.6",
    "--top_p", "0.95",
    "--do_sample",
    "--ans",
    "--dtype", "auto",
    "--outer_dtype", "auto",
    "--trust_remote_code", "1",
    "--enable_thinking", "0",
    "--agent1_model_name_or_path", PLANNER,
    "--agent2_model_name_or_path", CRITIC,
    "--agent3_model_name_or_path", SOLVER,
    "--agent1_inner_aligner_path", os.path.join(PLANNER, "adapter(math).pt"),
    "--agent2_inner_aligner_path", os.path.join(CRITIC, "adapter(math).pt"),
    "--agent3_inner_aligner_path", os.path.join(SOLVER, "adapter(math).pt"),
    "--outer_12_path", os.path.join(OUTER, "Planner-Critic-Outerlink(math).pt"),
    "--outer_23_path", os.path.join(OUTER, "Critic-Solver-Outerlink(math).pt"),
    "--outer_31_path", os.path.join(OUTER, "Solver-Planner-Outerlink(math).pt"),
    "--inner_adapter_type_fallback", "ln_res_adapter",
    "--outer_adapter_type_fallback", "outer_ln_res_adapter",
]

from inference_utils.inference_mas import main
main()
