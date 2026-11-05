"""
ADVERSARIAL STRESS TESTER & RED TEAM GENERATOR FOR IEEE MVC 2027
================================================================
Generates challenging test scenarios by:
1. Permuting event order based on benchmark proportions (DWPT, mu-drop, Crosswind).
2. Applying 10% - 20% domain randomization to parameters (wind, mu, timings, vehicle mass).
3. Superposing worst-case combinations (e.g. crosswind WHILE on ice, mu-drop inside DWPT).
4. Running an Adversarial Search to discover the weakest failure modes of our Hybrid RL system.
"""

import os
import sys
import time
import multiprocessing as mp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_rl_env import HybridKanonEnv
from train_hybrid_ppo import ActorCritic


class AdversarialScenario:
    def __init__(self, name, desc, **kwargs):
        self.name = name
        self.desc = desc
        
        # Benchmark defaults
        self.Tsim = kwargs.get('Tsim', 120.0)
        self.v_max_1 = kwargs.get('v_max_1', 15.0)
        self.v_max_2 = kwargs.get('v_max_2', 5.0)
        
        # Friction zones
        self.flag_road = kwargs.get('flag_road', 1)
        self.mu_high = kwargs.get('mu_high', 0.85)
        self.mu_low = kwargs.get('mu_low', 0.25)
        self.L_in = kwargs.get('L_in', 12.5)
        self.L_out = kwargs.get('L_out', 42.5)
        self.mu_mid = kwargs.get('mu_mid', 0.60)
        self.M_in = kwargs.get('M_in', 1976.99)
        self.M_out = kwargs.get('M_out', 2011.03)
        
        # Wind disturbance
        self.flag_wind = kwargs.get('flag_wind', 1)
        self.wind_start = kwargs.get('wind_start', 45.0)
        self.wind_stop = kwargs.get('wind_stop', 55.0)
        self.wind_force = kwargs.get('wind_force', 1600.0)
        self.wind_moment = kwargs.get('wind_moment', 160.0)
        self.wind_dir = kwargs.get('wind_dir', 1.0)
        self.wind_reversal = kwargs.get('wind_reversal', False) # Sudden flip mid-way
        
        # DWPT zone
        self.dwpt_x_start = kwargs.get('dwpt_x_start', 500.0)
        self.dwpt_x_end = kwargs.get('dwpt_x_end', 1500.0)
        
        # Physical variation
        self.mass_scale = kwargs.get('mass_scale', 1.0)


def build_curated_adversarial_suite():
    suite = []
    
    # 1. Superposed Gale-Ice Storm: Gió giật 1800N ĐÚNG LÚC XE ĐANG TRÊN BĂNG TUYẾT (t = 3.5 - 6.5s)
    suite.append(AdversarialScenario(
        name="ADV-01: Gale on Ice (Superposed)",
        desc="Crosswind 1800N hits exactly when vehicle is in the low-mu zone (3.5-6.5s)",
        wind_start=3.2,
        wind_stop=6.2,
        wind_force=1800.0,
        wind_moment=180.0,
        mu_low=0.20, # -20% friction
        L_in=12.5,
        L_out=45.0
    ))
    
    # 2. DWPT Slippery Trap: Vùng trơn low-mu bị đẩy vào BÊN TRONG vùng sạc DWPT (X = 700 - 750m) ở tốc độ 20 m/s
    suite.append(AdversarialScenario(
        name="ADV-02: Low-mu inside DWPT",
        desc="Low-mu ice patch (mu=0.22, 50m) located right inside DWPT zone at 72 km/h",
        L_in=700.0,
        L_out=750.0,
        mu_low=0.22,
        wind_start=45.0,
        wind_stop=55.0, # wind is at 642-842m, so low-mu is right inside wind too!
        wind_force=1600.0
    ))
    
    # 3. Wind Direction Sudden Reversal: Gió đổi hướng 180 độ (+1600N -> -1600N) lúc t=50s
    suite.append(AdversarialScenario(
        name="ADV-03: Sudden Wind Flip (+/-1600N)",
        desc="Crosswind violently reverses direction from +1600N to -1600N at t=50s",
        wind_start=45.0,
        wind_stop=55.0,
        wind_force=1600.0,
        wind_moment=160.0,
        wind_reversal=True
    ))
    
    # 4. Heavy Loaded Vehicle on Ultra-low mu: Xe chở tải nặng (+15% khối lượng = 1012 kg) trên mu=0.18
    suite.append(AdversarialScenario(
        name="ADV-04: Overweight Vehicle on Glaze Ice",
        desc="Vehicle mass +15% (1012kg) with ultra-low mu=0.18 and wind=1750N",
        mass_scale=1.15,
        mu_low=0.18,
        wind_force=1750.0,
        wind_moment=175.0,
        wind_start=45.0,
        wind_stop=55.0
    ))
    
    # 5. High-Speed Emergency Low-mu without Wind: Vùng trượt dài 60m ở tốc độ tối đa 20 m/s (X = 1600 - 1660m)
    suite.append(AdversarialScenario(
        name="ADV-05: High-Speed Hydroplaning (72 km/h)",
        desc="60m low-mu patch (mu=0.20) at cruise speed 20 m/s just after DWPT",
        L_in=1600.0,
        L_out=1660.0,
        mu_low=0.20,
        flag_wind=0
    ))
    
    # 6. Braking Zone Failure: Vùng phanh cuối chu trình bị trơn trượt cực nặng (mu=0.20 thay vì 0.60)
    suite.append(AdversarialScenario(
        name="ADV-06: Critical Braking Slip (mu=0.20)",
        desc="Final braking friction drops to mu=0.20 instead of 0.60 during hard stop",
        mu_mid=0.20,
        M_in=1960.0,
        M_out=2015.0,
        mu_low=0.25,
        L_in=12.5,
        L_out=42.5
    ))
    
    return suite


def evaluate_adversarial_scenario(args):
    scenario, model_path = args
    model = ActorCritic(state_dim=10, action_dim=3, hidden_dim=64)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    
    env = HybridKanonEnv(dt_macro=0.02, dt_micro=0.001)
    
    # Custom physical overrides
    if scenario.mass_scale != 1.0:
        env.sim.M *= scenario.mass_scale
        env.sim.I *= scenario.mass_scale
        env.sim.Nf *= scenario.mass_scale
        env.sim.Nr *= scenario.mass_scale
        
    def custom_road_friction(x, flag_road=1):
        if not scenario.flag_road: return scenario.mu_high
        if scenario.L_in <= x <= scenario.L_out: return scenario.mu_low
        elif scenario.M_in <= x <= scenario.M_out: return scenario.mu_mid
        return scenario.mu_high
        
    def custom_wind(t, flag_wind=1):
        if scenario.flag_wind and scenario.wind_start <= t <= scenario.wind_stop:
            f = scenario.wind_force
            m = scenario.wind_moment
            if scenario.wind_reversal and t >= (scenario.wind_start + scenario.wind_stop) / 2.0:
                f = -f
                m = -m
            return scenario.wind_dir * f, scenario.wind_dir * m
        return 0.0, 0.0
        
    def custom_vref(t):
        r1 = np.clip(2.5 * max(0.0, t), -scenario.v_max_1, scenario.v_max_1)
        r2 = np.clip(1.0 * max(0.0, t - 40.0), -scenario.v_max_2, scenario.v_max_2)
        r3 = np.clip(2.5 * max(0.0, t - (scenario.Tsim - 10.0)), -(scenario.v_max_1+scenario.v_max_2), (scenario.v_max_1+scenario.v_max_2))
        return max(0.0, r1 + r2 - r3)
        
    orig_dwpt = env.sim.get_dwpt_eta
    def custom_dwpt_eta(x, y):
        if x < scenario.dwpt_x_start or x > scenario.dwpt_x_end: return 0.0
        return orig_dwpt(x, y)
        
    env.sim.get_road_friction = custom_road_friction
    env.sim.get_wind = custom_wind
    env.sim.get_vref = custom_vref
    env.sim.get_dwpt_eta = custom_dwpt_eta
    env.Tsim = scenario.Tsim
    
    obs = env.reset(condition_id=1)
    done = False
    max_y = 0.0
    max_gamma = 0.0
    max_slip = 0.0
    
    while not done:
        action, _, _ = model.get_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if abs(info['PosY']) > max_y: max_y = abs(info['PosY'])
        if abs(env.gamma) > max_gamma: max_gamma = abs(env.gamma)
        
        # Check wheel slip excess
        speeds = env.w_wheels * env.R
        excess = np.max(speeds - env.Vx) if env.Vx > 0.1 else 0.0
        if excess > max_slip: max_slip = excess
        
    return {
        'name': scenario.name,
        'desc': scenario.desc,
        'is_dq': info['is_dq'],
        'score': info['total_Phi'],
        'p1': info['cum_Phi1'],
        'p2': info['cum_Phi2'],
        'p3': info['cum_Phi3'],
        'p4': info['cum_Phi4'],
        'max_y': max_y,
        'max_gamma': max_gamma,
        'max_slip': max_slip
    }


def run_adversarial_suite():
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, "best_hybrid_ppo.pt")
        
    suite = build_curated_adversarial_suite()
    args = [(s, model_path) for s in suite]
    
    workers = min(mp.cpu_count(), 6)
    with mp.Pool(processes=workers) as pool:
        results = pool.map(evaluate_adversarial_scenario, args)
        
    return results


def run_random_adversarial_search(n_samples=15):
    """
    Stochastic Red-Team Search: Randomly perturb parameters within 10%-20%
    and permute event orders to identify the worst-case score spikes.
    """
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(save_dir, "best_hybrid_ppo.pt")
        
    rng = np.random.RandomState(42)
    suite = []
    
    for i in range(n_samples):
        # 10% - 20% domain variation
        scale_wind = rng.uniform(0.85, 1.25)
        wind_f = 1600.0 * scale_wind
        wind_t_start = rng.uniform(10.0, 70.0) # permute wind time!
        wind_dur = rng.uniform(8.0, 12.0)
        
        scale_mu = rng.uniform(0.75, 1.20)
        mu_l = np.clip(0.25 * scale_mu, 0.16, 0.35)
        mu_x_start = rng.uniform(10.0, 900.0) # permute low-mu location!
        mu_len = rng.uniform(25.0, 50.0)
        
        mass_s = rng.uniform(0.90, 1.15)
        
        s = AdversarialScenario(
            name=f"SEARCH-RED-{i+1:02d}",
            desc=f"Wind={wind_f:.0f}N@t={wind_t_start:.1f}s, mu={mu_l:.2f}@X={mu_x_start:.0f}m, Mass={mass_s*880:.0f}kg",
            wind_start=wind_t_start,
            wind_stop=wind_t_start + wind_dur,
            wind_force=wind_f,
            wind_moment=wind_f * 0.1,
            mu_low=mu_l,
            L_in=mu_x_start,
            L_out=mu_x_start + mu_len,
            mass_scale=mass_s
        )
        suite.append(s)
        
    args = [(s, model_path) for s in suite]
    workers = min(mp.cpu_count(), 6)
    with mp.Pool(processes=workers) as pool:
        results = pool.map(evaluate_adversarial_scenario, args)
        
    return results


def main():
    print("=" * 80)
    print("  ADVERSARIAL RED-TEAM STRESS TEST SUITE (10% - 20% PERTURBATION & PERMUTATION)")
    print("=" * 80)
    
    t0 = time.time()
    results_curated = run_adversarial_suite()
    
    print("\n--- PART 1: 6 TARGETED ADVERSARIAL FAILURE-MODE SCENARIOS ---")
    for r in results_curated:
        status = "PASSED" if not r['is_dq'] else "DISQUALIFIED"
        print(f"\n[{status}] {r['name']}")
        print(f"    Scenario: {r['desc']}")
        print(f"    Phi={r['score']:.2f} | Phi1(Energy)={r['p1']:.1f} Wh | 60*Phi2(Slip)={60*r['p2']:.2f} | 3*Phi3(DWPT)={3*r['p3']:.1f} | 10*Phi4(ay)={10*r['p4']:.1f}")
        print(f"    Critical: max|PosY| = {r['max_y']*1000:.1f} mm | max|gamma| = {r['max_gamma']:.3f} rad/s | max_slip_excess = {r['max_slip']:.3f} m/s")
        
    print("\n" + "=" * 80)
    print("--- PART 2: ADVERSARIAL RANDOM SEARCH (15 SAMPLES) ---")
    print("=" * 80)
    results_search = run_random_adversarial_search(n_samples=15)
    
    # Sort search results by penalty Phi (descending - worst cases first)
    results_search.sort(key=lambda x: x['score'], reverse=True)
    
    for r in results_search[:5]: # Top 5 worst cases
        status = "PASSED" if not r['is_dq'] else "DISQUALIFIED"
        print(f"[{status}] {r['name']}: Phi={r['score']:.2f} (max|y|={r['max_y']*1000:.1f}mm, 60*Phi2={60*r['p2']:.2f}, 3*Phi3={3*r['p3']:.1f})")
        print(f"    Config: {r['desc']}")
        
    all_results = results_curated + results_search
    all_scores = [r['score'] for r in all_results if not r['is_dq']]
    dq_count = sum(1 for r in all_results if r['is_dq'])
    
    print("\n" + "=" * 80)
    print("  ADVERSARIAL RED-TEAM SUMMARY")
    print(f"  Total adversarial scenarios tested: {len(all_results)}")
    print(f"  Disqualified scenarios (drift > 5m): {dq_count} / {len(all_results)}")
    if all_scores:
        print(f"  Min penalty Phi: {min(all_scores):.2f}")
        print(f"  Max penalty Phi: {max(all_scores):.2f}")
        print(f"  Mean penalty Phi: {np.mean(all_scores):.2f}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")
    print("=" * 80)

if __name__ == '__main__':
    mp.freeze_support()
    main()
