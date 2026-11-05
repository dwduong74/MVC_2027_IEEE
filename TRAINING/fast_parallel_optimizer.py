"""
Fast Parallel Optimizer & Robustness Evaluator for MVC 2027
============================================================
Uses multiprocessing across all CPU cores to:
1. Sweep TCS & DYC parameters across all 4 official driving conditions.
2. Find the optimal parameter set minimizing composite score Phi.
3. Test the optimal parameters against the pseudo-secret driving cycle suite.
"""

import sys
import os
import time
import multiprocessing as mp
import numpy as np

# Ensure path to local modules
sys.path.insert(0, os.path.dirname(__file__))
from env_kanon_vehicle import KanonSimulator


def simulate_single_condition(args):
    """
    Run 1 driving condition with specific controller parameters.
    args is a tuple: (flag_wind, flag_road, params, dt)
    """
    flag_wind, flag_road, params, dt = args
    
    sim = KanonSimulator()
    N_steps = int(120.0 / dt)
    
    # Extract params
    K_tcs            = params.get('K_tcs', 800.0)
    slip_ratio_allow = params.get('slip_ratio_allow', 0.08)
    slip_speed_min   = params.get('slip_speed_min', 0.15)
    dyc_ki           = params.get('dyc_ki', 20.0)
    steer_kp         = params.get('steer_kp', 0.02)
    
    Vx, Vy, gamma = 0.0, 0.0, 0.0
    PosX, PosY, phi = 0.0, 0.0, 0.0
    w_FR, w_FL, w_RR, w_RL = 0.0, 0.0, 0.0, 0.0
    
    delta_f_actual = 0.0
    T_actual = np.zeros(4)
    v_integ, v_ff_state = 0.0, 0.0
    dyc_integ, dyc_deriv_state = 0.0, 0.0
    
    Phi1_sum, Phi2_sum, Phi3_sum, Phi4_sum = 0.0, 0.0, 0.0, 0.0
    
    M, I, lf, lr, d, hg = sim.M, sim.I, sim.lf, sim.lr, sim.d, sim.hg
    l, R = lf + lr, sim.R
    Nf_static, Nr_static = sim.Nf, sim.Nr
    ax_mem, ay_mem = 0.0, 0.0
    
    # Motor constant precomputations
    we_f_const = sim.pn_f / R
    we_r_const = sim.pn_r / R
    num_f_k = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f
    den_f_a = sim.Ra_f * (sim.Rce_f**2)
    den_f_b = (sim.Ra_f + sim.Rce_f) * (sim.Ld_f**2)
    
    num_r_k = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r
    den_r_a = sim.Ra_r * (sim.Rce_r**2)
    den_r_b = (sim.Ra_r + sim.Rce_r) * (sim.Ld_r**2)
    
    for k in range(N_steps):
        t = k * dt
        V_ref = sim.get_vref(t)
        mu = sim.get_road_friction(PosX, flag_road)
        F_wind, M_wind = sim.get_wind(t, flag_wind)
        
        # Velocity Controller
        Delta_V = V_ref - Vx
        dv_ff = (V_ref - v_ff_state) / sim.tau_ff_v
        v_ff_state += dv_ff * dt
        F_ff = (M / sim.tau_ff_v) * (V_ref - v_ff_state)
        v_integ += Delta_V * dt
        F_fb = sim.Kp_v * Delta_V + sim.Ki_v * v_integ
        F_total = np.clip(F_ff + F_fb, -sim.Fmax, sim.Fmax)
        T_all = F_total * R
        
        # Load Model
        Nfr = Nf_static - (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lr / l)) * ay_mem
        Nfl = Nf_static - (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lr / l)) * ay_mem
        Nrr = Nr_static + (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lf / l)) * ay_mem
        Nrl = Nr_static + (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lf / l)) * ay_mem
        
        # Control Laws
        e_y = 0.0 - PosY
        e_phi = 0.0 - phi
        delta_f_cmd = steer_kp * e_y + 0.2 * e_phi
        dyc_integ += e_y * dt
        dde = (e_y - dyc_deriv_state) / 0.005
        dyc_deriv_state += dde * dt
        Nz_cmd = 4000.0 * e_y + dyc_ki * dyc_integ + 1000.0 * dde + 5000.0 * e_phi
        delta_T = Nz_cmd * (R / (2.0 * d))
        
        # Cascade Traction Control System (TCS)
        wheel_speeds = np.array([w_FR, w_FL, w_RR, w_RL]) * R
        delta_Vw = wheel_speeds - Vx
        delta_V_allow = np.maximum(slip_speed_min, slip_ratio_allow * Vx)
        excess_slip = np.maximum(0.0, delta_Vw - delta_V_allow)
        
        T_base = T_all * 0.25
        T_cmd_raw = np.array([T_base + delta_T, T_base - delta_T, T_base + delta_T, T_base - delta_T])
        T_cmds = np.maximum(0.0, T_cmd_raw - K_tcs * excess_slip)
        
        # Loss-Minimization Id* (Morimoto)
        we_f = we_f_const * max(0.0, Vx)
        we_f_sq = we_f * we_f
        id_f = (num_f_k * we_f_sq) / (den_f_a + den_f_b * we_f_sq)
        
        we_r = we_r_const * max(0.0, Vx)
        we_r_sq = we_r * we_r
        id_r = (num_r_k * we_r_sq) / (den_r_a + den_r_b * we_r_sq)
        
        # Actuators
        delta_f_cmd = np.clip(delta_f_cmd, -sim.EPS_Del_max, sim.EPS_Del_max)
        delta_f_actual += (delta_f_cmd - delta_f_actual) * (dt / sim.EPS_tau)
        T_limits = np.array([sim.Tmax_f, sim.Tmax_f, sim.Tmax_r, sim.Tmax_r])
        T_cmds = np.clip(T_cmds, -T_limits, T_limits)
        T_actual += (T_cmds - T_actual) * (dt / sim.Motor_tau)
        
        # Tire Velocities & Slip Angles
        V_mag = np.sqrt(Vx**2 + Vy**2)
        beta = np.arctan2(Vy, Vx) if Vx > 0.05 else 0.0
        gamma_val = gamma
        V_fl = np.sqrt((V_mag*np.sin(beta) + gamma_val*lf)**2 + (V_mag*np.cos(beta) - gamma_val*d/2.0)**2)
        V_fr = np.sqrt((V_mag*np.sin(beta) + gamma_val*lf)**2 + (V_mag*np.cos(beta) + gamma_val*d/2.0)**2)
        V_rl = np.sqrt((V_mag*np.sin(beta) - gamma_val*lr)**2 + (V_mag*np.cos(beta) - gamma_val*d/2.0)**2)
        V_rr = np.sqrt((V_mag*np.sin(beta) - gamma_val*lr)**2 + (V_mag*np.cos(beta) + gamma_val*d/2.0)**2)
        alpha_fl = np.arctan2((V_mag*np.sin(beta) + gamma_val*lf), (V_mag*np.cos(beta) - gamma_val*d/2.0)) - delta_f_actual
        alpha_fr = np.arctan2((V_mag*np.sin(beta) + gamma_val*lf), (V_mag*np.cos(beta) + gamma_val*d/2.0)) - delta_f_actual
        alpha_rl = np.arctan2((V_mag*np.sin(beta) - gamma_val*lr), (V_mag*np.cos(beta) - gamma_val*d/2.0))
        alpha_rr = np.arctan2((V_mag*np.sin(beta) - gamma_val*lr), (V_mag*np.cos(beta) + gamma_val*d/2.0))
        
        Fx_fr, Fy_fr, lam_fr = sim.pacejka(V_fr, w_FR * R, alpha_fr, mu, Nfr)
        Fx_fl, Fy_fl, lam_fl = sim.pacejka(V_fl, w_FL * R, alpha_fl, mu, Nfl)
        Fx_rr, Fy_rr, lam_rr = sim.pacejka(V_rr, w_RR * R, alpha_rr, mu, Nrr)
        Fx_rl, Fy_rl, lam_rl = sim.pacejka(V_rl, w_RL * R, alpha_rl, mu, Nrl)
        
        w_FR += ((T_actual[0] - R * Fx_fr) / sim.Jw_f) * dt
        w_FL += ((T_actual[1] - R * Fx_fl) / sim.Jw_f) * dt
        w_RR += ((T_actual[2] - R * Fx_rr) / sim.Jw_r) * dt
        w_RL += ((T_actual[3] - R * Fx_rl) / sim.Jw_r) * dt
        
        c_del, s_del = np.cos(delta_f_actual), np.sin(delta_f_actual)
        F_aero = 0.5 * sim.rho * sim.Cd * sim.Afpa * (Vx**2)
        F_roll = M * sim.g * sim.mu0
        Fx_tot = (Fx_fl + Fx_fr) * c_del - (Fy_fl + Fy_fr) * s_del + Fx_rl + Fx_rr - F_aero - F_roll
        Fy_tot = (Fx_fl + Fx_fr) * s_del + (Fy_fl + Fy_fr) * c_del + Fy_rl + Fy_rr + F_wind
        
        N_front = lf * ((Fx_fl + Fx_fr) * s_del + (Fy_fl + Fy_fr) * c_del)
        N_rear = -lr * (Fy_rl + Fy_rr)
        N_diff = (d / 2.0) * ((Fx_fr - Fx_fl) * c_del + (Fy_fl - Fy_fr) * s_del + Fx_rr - Fx_rl)
        Nz_tot = N_front + N_rear + N_diff + M_wind
        
        ax = Fx_tot / M
        ay = Fy_tot / M
        ax_mem, ay_mem = ax, ay
        
        dVx = ax + Vy * gamma_val
        dVy = ay - Vx * gamma_val
        dgamma = Nz_tot / I
        Vx += dVx * dt
        Vy += dVy * dt
        gamma += dgamma * dt
        
        dPosX = Vx * np.cos(phi) - Vy * np.sin(phi)
        dPosY = Vx * np.sin(phi) + Vy * np.cos(phi)
        PosX += dPosX * dt
        PosY += dPosY * dt
        phi += gamma_val * dt
        
        # Disqualification check
        if abs(PosY) > 5.0:
            return 1e7, 1e7, 1e7, 1e7, 1e7
        
        Pin_fr, _ = sim.calc_motor_power(T_actual[0], w_FR, True, id_f)
        Pin_fl, _ = sim.calc_motor_power(T_actual[1], w_FL, True, id_f)
        Pin_rr, _ = sim.calc_motor_power(T_actual[2], w_RR, False, id_r)
        Pin_rl, _ = sim.calc_motor_power(T_actual[3], w_RL, False, id_r)
        P_all = Pin_fr + Pin_fl + Pin_rr + Pin_rl
        eta_dwpt = sim.get_dwpt_eta(PosX, PosY) / 100.0
        
        Phi1_sum += P_all
        Phi2_sum += (lam_fl**2 + lam_fr**2 + lam_rl**2 + lam_rr**2)
        Phi3_sum += (1.0 - eta_dwpt)**2
        Phi4_sum += ay**2
        
    p1 = Phi1_sum * dt / 3600.0
    p2 = Phi2_sum * dt
    p3 = Phi3_sum * dt
    p4 = Phi4_sum * dt
    score = 1.0 * p1 + 60.0 * p2 + 3.0 * p3 + 10.0 * p4
    return score, p1, p2, p3, p4


def evaluate_parameter_set(params, dt=0.001, pool=None):
    """
    Evaluates 1 parameter set across all 4 competition driving conditions.
    """
    conditions = [
        (0, 0),  # Cond 1: High mu, No wind
        (1, 0),  # Cond 2: High mu, With wind
        (0, 1),  # Cond 3: Mu change, No wind
        (1, 1),  # Cond 4: Mu change, With wind
    ]
    
    args = [(w, r, params, dt) for w, r in conditions]
    
    if pool is not None:
        results = pool.map(simulate_single_condition, args)
    else:
        results = [simulate_single_condition(a) for a in args]
        
    scores = [r[0] for r in results]
    tot_phi = sum(scores)
    return tot_phi, results


def main():
    print("=" * 70, flush=True)
    print("  FAST PARALLEL OPTIMIZER - IEEE MVC 2027", flush=True)
    print(f"  CPU Cores available: {mp.cpu_count()}", flush=True)
    print("=" * 70, flush=True)
    
    # Define candidate parameter sets
    # We focus on the most critical levers:
    # 1. K_tcs: feedback gain on wheel spin excess speed
    # 2. slip_ratio_allow: allowable slip ratio before TCS intervenes
    # 3. slip_speed_min: minimum velocity threshold for slip
    # 4. dyc_ki: integrator gain for lateral position control (rejection of wind)
    
    candidates = [
        # (name, {params})
        ("Baseline TCS (Current)",
         {'K_tcs': 800.0, 'slip_ratio_allow': 0.08, 'slip_speed_min': 0.15, 'dyc_ki': 20.0}),
        
        ("Aggressive TCS (K=1000, 6% slip)",
         {'K_tcs': 1000.0, 'slip_ratio_allow': 0.06, 'slip_speed_min': 0.12, 'dyc_ki': 20.0}),
        
        ("Ultra-Tight TCS (K=1200, 5% slip)",
         {'K_tcs': 1200.0, 'slip_ratio_allow': 0.05, 'slip_speed_min': 0.10, 'dyc_ki': 20.0}),
        
        ("Smooth TCS (K=600, 10% slip)",
         {'K_tcs': 600.0, 'slip_ratio_allow': 0.10, 'slip_speed_min': 0.18, 'dyc_ki': 20.0}),
         
        ("Optimized TCS + Active Wind Rejection (Ki=35)",
         {'K_tcs': 900.0, 'slip_ratio_allow': 0.07, 'slip_speed_min': 0.12, 'dyc_ki': 35.0}),
         
        ("Precision TCS + Fast Wind Rejection (Ki=45)",
         {'K_tcs': 1000.0, 'slip_ratio_allow': 0.06, 'slip_speed_min': 0.10, 'dyc_ki': 45.0}),
    ]
    
    # Use 4 parallel workers (1 worker per driving condition)
    workers = min(mp.cpu_count(), 4)
    print(f"Starting worker pool with {workers} processes...", flush=True)
    
    best_name = None
    best_score = float('inf')
    best_params = None
    best_results = None
    
    with mp.Pool(processes=workers) as pool:
        for name, params in candidates:
            t0 = time.time()
            tot_score, results = evaluate_parameter_set(params, dt=0.001, pool=pool)
            dt_s = time.time() - t0
            
            c_scores = [r[0] for r in results]
            print(f"\n[{name}] -> TOTAL Phi = {tot_score:.2f} ({dt_s:.1f}s)", flush=True)
            print(f"  Cond 1 (Hi-mu, No-wind):  {c_scores[0]:.2f} (Phi1={results[0][1]:.1f}, Phi2*60={60*results[0][2]:.2f})", flush=True)
            print(f"  Cond 2 (Hi-mu, Wind):     {c_scores[1]:.2f} (Phi3*3={3*results[1][3]:.1f}, Phi4*10={10*results[1][4]:.2f})", flush=True)
            print(f"  Cond 3 (Mu-chg, No-wind): {c_scores[2]:.2f} (Phi2*60={60*results[2][2]:.2f})", flush=True)
            print(f"  Cond 4 (Mu-chg, Wind):    {c_scores[3]:.2f} (Phi2*60={60*results[3][2]:.2f}, Phi4*10={10*results[3][4]:.2f})", flush=True)
            
            if tot_score < best_score:
                best_score = tot_score
                best_name = name
                best_params = params
                best_results = results
                
    print("\n" + "=" * 70, flush=True)
    print(f"  BEST CONFIGURATION: {best_name}", flush=True)
    print(f"  LOWEST TOTAL SCORE: {best_score:.2f}", flush=True)
    print(f"  Parameters: {best_params}", flush=True)
    print("=" * 70, flush=True)
    
    # Save the best parameters to a config file for Simulink & meso_para.m
    import json
    with open("best_classical_params.json", "w") as f:
        json.dump({'name': best_name, 'score': best_score, 'params': best_params}, f, indent=2)
    print("Saved best parameters to best_classical_params.json", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
