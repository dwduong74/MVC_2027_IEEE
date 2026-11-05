import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_rl_env import HybridKanonEnv
from train_hybrid_ppo import ActorCritic
from adversarial_stress_tester import build_curated_adversarial_suite, evaluate_adversarial_scenario

save_dir = os.path.dirname(__file__)
model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")
suite = build_curated_adversarial_suite()

# Evaluate ADV-01 and ADV-02 baseline
print("Testing ADV-01 and ADV-02 on current code:")
for s in suite[:2]:
    r = evaluate_adversarial_scenario((s, model_path))
    status = "PASSED" if not r['is_dq'] else "DISQUALIFIED"
    print(f"[{status}] {r['name']}: Phi={r['score']:.2f}, max|y|={r['max_y']*1000:.1f}mm, 10*Phi4={10*r['p4']:.1f}, slip={r['max_slip']:.3f}m/s")
