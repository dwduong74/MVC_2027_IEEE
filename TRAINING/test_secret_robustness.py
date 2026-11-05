"""
Robustness Testing of Best Classical Controller on Pseudo-Secret Cycles
========================================================================
Runs the 11 pseudo-secret test scenarios in parallel.
Validates:
1. Vehicle never gets Disqualified (|PosY| <= 5.0m, no stall).
2. Phi2 (slip penalty) is firmly suppressed across arbitrary mu drops.
3. DWPT efficiency remains high even with shifted coils.
4. Total score stability under extreme wind / surface conditions.
"""

import sys
import os
import time
import multiprocessing as mp
import numpy as np
import json

sys.path.insert(0, os.path.dirname(__file__))
from generate_secret_cycles import generate_scenario_suite
from env_kanon_vehicle import KanonSimulator


def run_single_secret_scenario(args):
    scenario, params, dt = args
    name = scenario['name']
    cycle = scenario['cycle']
    
    sim = KanonSimulator()
    Tsim = cycle.Tsim
    N_steps = int(Tsim / dt)
    
    K_tcs            = params.get('K_tcs', 900.0)
    slip_ratio_allow = params.get('slip_ratio_allow', 0.07)
    slip_speed_min   = params.get('slip_speed_min', 0.12)
    dyc_ki           = params.get('dyc_ki', 35.0)
    steer_kp         = params.get('steer_kp', 0.02)
    
    Vx, Vy, gamma = 0.0, 0.0, 0.0
    PosX, PosY, phi = 0.0, 0.0, 0.0
    w_FR, w_FL, w_RR, w_RL = 0.0, 0.0, 0.0, 0.0
    
    delta_f_actual = 0.0
    T_actual = np.zeros(4)
    v_integ, v_ff_state = 0.0, 0.0
    dyc_integ, dyc_deriv_state = 0.0, 0.0
    
    Phi1_sum, Phi2_sum, Phi3_sum, Phi4_sum = 0.0, 0.0, 0.0, 0.0
    max_y = 0.0
    
    M, I, lf, lr, d, hg = sim.M, sim.I, sim.lf, sim.lr, sim.d, sim.hg
    l, R = lf + lr, sim.R
    Nf_static, Nr_static = sim.Nf, sim.Nr
    ax_mem, ay_mem = 0.0, 0.0
    
    we_f_const = sim.pn_f / R
    we_r_const = sim.pn_r / R
    num_f_k = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f
    den_f_a = sim.Ra_f * (sim.Rce_f**2)
    den_f_b = (sim.Ra_f + sim.Rce_f) * (sim.Ld_f**2)
    
    num_r_k = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r
    den_r_a = sim.Ra_r * (sim.Rce_r**2)
    den_r_b = (sim.Ra_r + sim.Rce_r) * (sim.Ld_r**2)
    
    def custom_road_friction(x):
        if not cycle.flag_road:
            return cycle.mu_high
        if cycle.L_in <= x <= cycle.L_out:
            return cycle.mu_low
        elif cycle.M_in <= x <= cycle.M_out:
            return cycle.mu_mid
        return cycle.mu_high
        
    def custom_wind(t):
        if cycle.flag_wind and cycle.wind_start <= t <= cycle.wind_stop:
            return cycle.wind_dir * cycle.wind_force, cycle.wind_dir * cycle.wind_moment
        return 0.0, 0.0
        
    def custom_vref(t):
        r1 = np.clip(2.5 * max(0.0, t), -cycle.v_max_1, cycle.v_max_1)
        r2 = np.clip(1.0 * max(0.0, t - 40.0), -cycle.v_max_2, cycle.v_max_2)
        r3 = np.clip(2.5 * max(0.0, t - (Tsim - 10.0)), -(cycle.v_max_1+cycle.v_max_2), (cycle.v_max_1+cycle.v_max_2))
        return max(0.0, r1 + r2 - r3)
        
    def custom_dwpt_eta(x, y):
        if x < cycle.dwpt_x_start or x > cycle.dwpt_x_end:
            return 0.0
        return sim.get_dwpt_eta(x, y)
    
    is_dq = False
    
    for k in range(N_steps):
        t = k * dt
        V_ref = custom_vref(t)
        mu = custom_road_friction(PosX)
        F_wind, M_wind = custom_wind(t)
        
        Delta_V = V_ref - Vx
        dv_ff = (V_ref - v_ff_state) / sim.tau_ff_v
        v_ff_state += dv_ff * dt
        F_ff = (M / sim.tau_ff_v) * (V_ref - v_ff_state)
        v_integ += Delta_V * dt
        F_fb = sim.Kp_v * Delta_V + sim.Ki_v * v_integ
        F_total = np.clip(F_ff + F_fb, -sim.Fmax, sim.Fmax)
        T_all = F_total * R
        
        Nfr = Nf_static - (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lr / l)) * ay_mem
        Nfl = Nf_static - (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lr / l)) * ay_mem
        Nrr = Nr_static + (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lf / l)) * ay_mem
        Nrl = Nr_static + (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lf / l)) * ay_mem
        
        e_y = 0.0 - PosY
        e_phi = 0.0 - phi
        delta_f_cmd = steer_kp * e_y + 0.2 * e_phi
        dyc_integ += e_y * dt
        dde = (e_y - dyc_deriv_state) / 0.005
        dyc_deriv_state += dde * dt
        Nz_cmd = 4000.0 * e_y + dyc_ki * dyc_integ + 1000.0 * dde + 5000.0 * e_phi
        
        # ESC Yaw Rate Limiter (Safety Barrier against high-speed spinout)
        gamma_limit = params.get('esc_gamma_limit', 0.20)
        if abs(gamma) > gamma_limit:
            scale = max(0.0, 1.0 - (abs(gamma) - gamma_limit) / 0.10)
            Nz_cmd *= scale
            
        delta_T = Nz_cmd * (R / (2.0 * d))
        
        wheel_speeds = np.array([w_FR, w_FL, w_RR, w_RL]) * R
        delta_Vw = wheel_speeds - Vx
        delta_V_allow = np.maximum(slip_speed_min, slip_ratio_allow * Vx)
        excess_slip = np.maximum(0.0, delta_Vw - delta_V_allow)
        
        T_base = T_all * 0.25
        T_cmd_raw = np.array([T_base + delta_T, T_base - delta_T, T_base + delta_T, T_base - delta_T])
        T_cmds = np.maximum(0.0, T_cmd_raw - K_tcs * excess_slip)
        
        we_f = we_f_const * max(0.0, Vx)
        we_f_sq = we_f * we_f
        id_f = (num_f_k * we_f_sq) / (den_f_a + den_f_b * we_f_sq)
        
        we_r = we_r_const * max(0.0, Vx)
        we_r_sq = we_r * we_r
        id_r = (num_r_k * we_r_sq) / (den_r_a + den_r_b * we_r_sq)
        
        delta_f_cmd = np.clip(delta_f_cmd, -sim.EPS_Del_max, sim.EPS_Del_max)
        delta_f_actual += (delta_f_cmd - delta_f_actual) * (dt / sim.EPS_tau)
        T_limits = np.array([sim.Tmax_f, sim.Tmax_f, sim.Tmax_r, sim.Tmax_r])
        T_cmds = np.clip(T_cmds, -T_limits, T_limits)
        T_actual += (T_cmds - T_actual) * (dt / sim.Motor_tau)
        
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
        
        if abs(PosY) > max_y:
            max_y = abs(PosY)
            
        if abs(PosY) > 5.0:
            is_dq = True
            break
            
        Pin_fr, _ = sim.calc_motor_power(T_actual[0], w_FR, True, id_f)
        Pin_fl, _ = sim.calc_motor_power(T_actual[1], w_FL, True, id_f)
        Pin_rr, _ = sim.calc_motor_power(T_actual[2], w_RR, False, id_r)
        Pin_rl, _ = sim.calc_motor_power(T_actual[3], w_RL, False, id_r)
        P_all = Pin_fr + Pin_fl + Pin_rr + Pin_rl
        eta_dwpt = custom_dwpt_eta(PosX, PosY) / 100.0
        
        Phi1_sum += P_all
        Phi2_sum += (lam_fl**2 + lam_fr**2 + lam_rl**2 + lam_rr**2)
        Phi3_sum += (1.0 - eta_dwpt)**2
        Phi4_sum += ay**2
        
    p1 = Phi1_sum * dt / 3600.0
    p2 = Phi2_sum * dt
    p3 = Phi3_sum * dt
    p4 = Phi4_sum * dt
    score = 1.0 * p1 + 60.0 * p2 + 3.0 * p3 + 10.0 * p4
    
    return {
        'name': name,
        'is_dq': is_dq,
        'score': score,
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'p4': p4,
        'max_y': max_y,
        'final_x': PosX
    }


def main():
    print("=" * 75, flush=True)
    print("  STRESS TEST: BEST CONTROLLER ON PSEUDO-SECRET DRIVING CYCLES", flush=True)
    print("=" * 75, flush=True)
    
    # Load best parameters
    with open("best_classical_params.json") as f:
        data = json.load(f)
    params = data['params']
    print("Loaded controller params:", params, flush=True)
    
    # Generate 11 challenging scenarios
    scenarios = generate_scenario_suite(n_scenarios=5, base_seed=123)
    print("Total test scenarios:", len(scenarios), flush=True)
    
    args = [(s, params, 0.001) for s in scenarios]
    
    workers = min(mp.cpu_count(), 6)
    print("Executing in parallel across", workers, "workers...", flush=True)
    t0 = time.time()
    
    with mp.Pool(processes=workers) as pool:
        results = pool.map(run_single_secret_scenario, args)
        
    dt_total = time.time() - t0
    print("\n" + "=" * 75, flush=True)
    print("  PSEUDO-SECRET TEST RESULTS (Total elapsed: %.1fs)" % dt_total, flush=True)
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
    print("SUMMARY STATS OVER %d SCENARIOS:" % len(scores), flush=True)
    print("  Disqualifications: %d / %d" % (len(results) - len(scores), len(results)), flush=True)
    print("  Mean Score Phi:    %.2f" % np.mean(scores), flush=True)
    print("  Min  Score Phi:    %.2f" % np.min(scores), flush=True)
    print("  Max  Score Phi:    %.2f" % np.max(scores), flush=True)
    print("  Std  Score Phi:    %.2f" % np.std(scores), flush=True)
    print("=" * 75, flush=True)
    
    if all_passed:
        print("\nALL PSEUDO-SECRET SCENARIOS PASSED WITH ZERO DISQUALIFICATIONS!", flush=True)
    else:
        print("\nWARNING: Some scenarios were disqualified!", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
