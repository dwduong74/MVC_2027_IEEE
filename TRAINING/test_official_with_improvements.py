import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from test_traction_derating_sim import simulate_with_controller
from adversarial_stress_tester import AdversarialScenario

def main():
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")

    conds = [
        AdversarialScenario(name="Cond 1", desc="mu=0.85, No Wind", flag_road=0, flag_wind=0),
        AdversarialScenario(name="Cond 2", desc="mu=0.85, Wind 1600N", flag_road=0, flag_wind=1, wind_start=45.0, wind_stop=55.0, wind_force=1600.0, wind_moment=160.0),
        AdversarialScenario(name="Cond 3", desc="mu-change, No Wind", flag_road=1, flag_wind=0, mu_low=0.25, L_in=12.5, L_out=42.5, mu_mid=0.60, M_in=1976.99, M_out=2011.03),
        AdversarialScenario(name="Cond 4", desc="mu-change, Wind 1600N", flag_road=1, flag_wind=1, mu_low=0.25, L_in=12.5, L_out=42.5, mu_mid=0.60, M_in=1976.99, M_out=2011.03, wind_start=45.0, wind_stop=55.0, wind_force=1600.0, wind_moment=160.0)
    ]

    print("Evaluating Official 4 Conditions with Improved Control:", flush=True)
    scores = []
    for c in conds:
        r = simulate_with_controller(c, model_path)
        scores.append(r['score'])
        print(f"{r['name']} ({c.desc}) -> Phi={r['score']:.2f} (Phi1={r['p1']:.1f}, Phi2*60={60*r['p2']:.2f}, Phi3*3={3*r['p3']:.1f}, Phi4*10={10*r['p4']:.1f}, max|y|={r['max_y']*1000:.1f}mm)", flush=True)

    print(f"\nTOTAL SCORE PHI: {sum(scores):.2f}", flush=True)

if __name__ == '__main__':
    main()
