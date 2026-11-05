"""
Evaluate Hybrid RL Policy on Pseudo-Secret Driving Cycles
==========================================================
Runs the 11 pseudo-secret test scenarios with the trained Hybrid RL Policy
to verify robustness, zero disqualifications, and overall performance.
"""

import sys
import os
import time
import multiprocessing as mp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from generate_secret_cycles import generate_scenario_suite
from train_hybrid_ppo import ActorCritic
from hybrid_rl_env import HybridKanonEnv


def run_hybrid_secret_scenario(args):
    scenario, model_path, dt_macro = args
    name = scenario['name']
    cycle = scenario['cycle']
    
    # Load model
    model = ActorCritic(state_dim=10, action_dim=3, hidden_dim=64)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        
    env = HybridKanonEnv(dt_macro=dt_macro, dt_micro=0.001)
    
    # Monkey patch env's simulator methods to use cycle's randomized parameters
    def custom_road_friction(x, flag_road=1):
        if not cycle.flag_road: return cycle.mu_high
        if cycle.L_in <= x <= cycle.L_out: return cycle.mu_low
        elif cycle.M_in <= x <= cycle.M_out: return cycle.mu_mid
        return cycle.mu_high
        
    def custom_wind(t, flag_wind=1):
        if cycle.flag_wind and cycle.wind_start <= t <= cycle.wind_stop:
            return cycle.wind_dir * cycle.wind_force, cycle.wind_dir * cycle.wind_moment
        return 0.0, 0.0
        
    def custom_vref(t):
        r1 = np.clip(2.5 * max(0.0, t), -cycle.v_max_1, cycle.v_max_1)
        r2 = np.clip(1.0 * max(0.0, t - 40.0), -cycle.v_max_2, cycle.v_max_2)
        r3 = np.clip(2.5 * max(0.0, t - (cycle.Tsim - 10.0)), -(cycle.v_max_1+cycle.v_max_2), (cycle.v_max_1+cycle.v_max_2))
        return max(0.0, r1 + r2 - r3)
        
    orig_dwpt = env.sim.get_dwpt_eta
    def custom_dwpt_eta(x, y):
        if x < cycle.dwpt_x_start or x > cycle.dwpt_x_end: return 0.0
        return orig_dwpt(x, y)
        
    env.sim.get_road_friction = custom_road_friction
    env.sim.get_wind = custom_wind
    env.sim.get_vref = custom_vref
    env.sim.get_dwpt_eta = custom_dwpt_eta
    env.Tsim = cycle.Tsim
    
    obs = env.reset(condition_id=1)
    done = False
    max_y = 0.0
    
    while not done:
        action, _, _ = model.get_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if abs(info['PosY']) > max_y:
            max_y = abs(info['PosY'])
            
    return {
        'name': name,
        'is_dq': info['is_dq'],
        'score': info['total_Phi'],
        'p1': info['cum_Phi1'],
        'p2': info['cum_Phi2'],
        'p3': info['cum_Phi3'],
        'p4': info['cum_Phi4'],
        'max_y': max_y
    }


def main():
    print("=" * 75, flush=True)
    print("  HYBRID RL POLICY STRESS TEST ON 11 PSEUDO-SECRET DRIVING CYCLES", flush=True)
    print("=" * 75, flush=True)
    
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, "best_hybrid_ppo.pt")
        
    scenarios = generate_scenario_suite(n_scenarios=5, base_seed=123)
    args = [(s, model_path, 0.02) for s in scenarios]
    
    workers = min(mp.cpu_count(), 6)
    print("Executing in parallel across %d workers..." % workers, flush=True)
    t0 = time.time()
    
    with mp.Pool(processes=workers) as pool:
        results = pool.map(run_hybrid_secret_scenario, args)
        
    dt_total = time.time() - t0
    print("\n" + "=" * 75, flush=True)
    print("  HYBRID RL PSEUDO-SECRET RESULTS (Elapsed: %.1fs)" % dt_total, flush=True)
    print("=" * 75, flush=True)
    
    all_passed = True
    scores = []
    
    for r in results:
        status = "PASSED" if not r['is_dq'] else "DISQUALIFIED"
        if r['is_dq']:
            all_passed = False
        else:
            scores.append(r['score'])
            
        print("[%-12s] %-42s -> Phi=%7.2f (Phi1=%5.1f, Phi2*60=%5.2f, Phi3*3=%5.1f, max|y|=%5.1fmm)" % (
            status, r['name'], r['score'], r['p1'], 60.0*r['p2'], 3.0*r['p3'], r['max_y']*1000.0
        ), flush=True)
        
    print("-" * 75, flush=True)
    print("SUMMARY STATS:")
    print("  Disqualifications: %d / %d" % (len(results) - len(scores), len(results)), flush=True)
    print("  Mean Score Phi:    %.2f" % np.mean(scores), flush=True)
    print("  Min  Score Phi:    %.2f" % np.min(scores), flush=True)
    print("  Max  Score Phi:    %.2f" % np.max(scores), flush=True)
    print("=" * 75, flush=True)
    
    if all_passed:
        print("\nALL 11 SCENARIOS PASSED WITH ZERO DISQUALIFICATIONS!", flush=True)
    else:
        print("\nWARNING: Some scenarios were disqualified!", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
