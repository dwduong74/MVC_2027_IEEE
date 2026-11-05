"""
Full Official IEEE MVC 2027 Benchmark Evaluation & Scoring
===========================================================
Scores all 3 strategies on the 4 official benchmark conditions:
  1. Baseline Strategy (BTC Default)
  2. Improved Strategy (BTC Provided)
  3. Our Proposed Upgraded Controller (Hybrid RL + Lateral-Priority TCS + ESC Damping)
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hybrid_rl_env import HybridKanonEnv
from train_hybrid_ppo import ActorCritic
from env_kanon_vehicle import KanonSimulator


def run_single_simulation(strategy_type, cond_id, dt=0.0005):
    """
    strategy_type:
      1: Baseline Strategy
      2: BTC Improved Strategy
      3: Our Upgraded Strategy (Hybrid RL + Lateral-Priority TCS + ESC Damping)
    cond_id: 1, 2, 3, 4
    """
    sim = KanonSimulator()
    Tsim = 120.0
    N_steps = int(round(Tsim / dt))
    
    flag_wind = 1 if cond_id in [2, 4] else 0
    flag_road = 1 if cond_id in [3, 4] else 0
    
    # Model for strategy 3
    if strategy_type == 3:
        save_dir = os.path.dirname(__file__)
        model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")
        if not os.path.exists(model_path):
            model_path = os.path.join(save_dir, "best_hybrid_ppo.pt")
        model = ActorCritic(state_dim=10, action_dim=3, hidden_dim=64)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        
    # State variables
    Vx, Vy, gamma = 0.0, 0.0, 0.0
    PosX, PosY, phi = 0.0, 0.0, 0.0
    w_wheels = np.zeros(4)
    delta_f_actual = 0.0
    T_actual = np.zeros(4)
    
    v_integ = 0.0
    v_ff_state = 0.0
    dyc_integ = 0.0
    dyc_deriv_state = 0.0
    Nz_filtered = 0.0
    
    Phi1_sum = 0.0
    Phi2_sum = 0.0
    Phi3_sum = 0.0
    Phi4_sum = 0.0
    
    M, I = sim.M, sim.I
    lf, lr, d, hg = sim.lf, sim.lr, sim.d, sim.hg
    l = lf + lr
    R = sim.R
    Nf_static, Nr_static = sim.Nf, sim.Nr
    ax_mem, ay_mem = 0.0, 0.0
    
    macro_dt = 0.02
    macro_steps = int(round(macro_dt / dt))
    alpha_throttle = 1.0
    r_front = 0.50
    delta_Nz_rl = 0.0
    
    max_y = 0.0
    
    for step in range(N_steps):
        t = step * dt
        V_ref = sim.get_vref(t)
        mu = sim.get_road_friction(PosX, flag_road)
        F_wind, M_wind = sim.get_wind(t, flag_wind)
        
        # Macro step for RL
        if strategy_type == 3 and (step % macro_steps == 0):
            # Compute observation
            v_err = V_ref - Vx
            speeds = w_wheels * R
            excess = np.max(speeds - Vx) if Vx > 0.1 else 0.0
            in_dwpt = 1.0 if (500.0 <= PosX <= 1500.0) else 0.0
            obs = np.array([
                Vx / 20.0,
                v_err / 5.0,
                np.clip(PosY / 0.10, -5.0, 5.0),
                np.clip(Vy / 2.0, -5.0, 5.0),
                np.clip(phi / 0.10, -5.0, 5.0),
                np.clip(gamma / 0.30, -5.0, 5.0),
                np.clip(ay_mem / 2.0, -5.0, 5.0),
                np.clip(excess / 0.15, 0.0, 5.0),
                in_dwpt,
                PosX / 2022.0
            ], dtype=np.float32)
            with torch.no_grad():
                act, _, _ = model.get_action(obs, deterministic=True)
            alpha_throttle = 1.0 + 0.20 * np.clip(act[0], -1.0, 1.0)
            r_front = 0.50 + 0.15 * np.clip(act[1], -1.0, 1.0)
            delta_Nz_rl = 500.0 * np.clip(act[2], -1.0, 1.0)
            
        # Velocity Controller
        Delta_V = V_ref - Vx
        dv_ff = (V_ref - v_ff_state) / sim.tau_ff_v
        v_ff_state += dv_ff * dt
        F_ff = (M / sim.tau_ff_v) * (V_ref - v_ff_state)
        v_integ += Delta_V * dt
        F_fb = sim.Kp_v * Delta_V + sim.Ki_v * v_integ
        F_total = np.clip(F_ff + F_fb, -sim.Fmax, sim.Fmax)
        
        if strategy_type == 3:
            T_all = F_total * R * alpha_throttle
            # Lateral-Priority Traction De-rating
            if T_all > 0:
                wheel_speeds = w_wheels * R
                current_excess = np.max(wheel_speeds - Vx) if Vx > 0.1 else 0.0
                slip_metric = current_excess / 0.12
                derate = 1.0 - np.clip(15.0 * (abs(PosY) - 0.015) + 20.0 * slip_metric * abs(PosY), 0.0, 1.0)
                T_all *= max(0.0, derate)
        else:
            T_all = F_total * R
            
        # Load Model
        Nfr = Nf_static - (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lr / l)) * ay_mem
        Nfl = Nf_static - (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lr / l)) * ay_mem
        Nrr = Nr_static + (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lf / l)) * ay_mem
        Nrl = Nr_static + (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lf / l)) * ay_mem
        
        e_y = 0.0 - PosY
        e_phi = 0.0 - phi
        
        if strategy_type == 1:
            # Baseline Strategy
            delta_f_cmd = 0.02 * e_y + 0.2 * e_phi
            T_FR_cmd = T_all * 0.27
            T_FL_cmd = T_all * 0.27
            T_RR_cmd = T_all * 0.23
            T_RL_cmd = T_all * 0.23
            Id_FR, Id_FL, Id_RR, Id_RL = 0.0, 0.0, 0.0, 0.0
            
        elif strategy_type == 2:
            # BTC Improved Strategy
            delta_f_cmd = 0.02 * e_y + 0.2 * e_phi
            dyc_integ += e_y * dt
            dde = (e_y - dyc_deriv_state) / 0.005
            dyc_deriv_state += dde * dt
            Nz_cmd = 4000.0 * e_y + 20.0 * dyc_integ + 1000.0 * dde + 5000.0 * e_phi
            delta_T = Nz_cmd * (R / (2.0 * d))
            
            T_FR_cmd = T_all * 0.25 + delta_T
            T_FL_cmd = T_all * 0.25 - delta_T
            T_RR_cmd = T_all * 0.25 + delta_T
            T_RL_cmd = T_all * 0.25 - delta_T
            
            we_f = sim.pn_f * max(0.0, Vx / R)
            num_f = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f * (we_f**2)
            den_f = sim.Ra_f * (sim.Rce_f**2) + (sim.Ra_f + sim.Rce_f) * ((sim.Ld_f * we_f)**2)
            id_f = num_f / den_f if den_f > 0 else 0.0
            
            we_r = sim.pn_r * max(0.0, Vx / R)
            num_r = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r * (we_r**2)
            den_r = sim.Ra_r * (sim.Rce_r**2) + (sim.Ra_r + sim.Rce_r) * ((sim.Ld_r * we_r)**2)
            id_r = num_r / den_r if den_r > 0 else 0.0
            Id_FR, Id_FL, Id_RR, Id_RL = id_f, id_f, id_r, id_r
            
        elif strategy_type == 3:
            # Our Upgraded Strategy
            delta_f_cmd = 0.02 * e_y + 0.20 * e_phi
            dyc_integ += e_y * dt
            dde = (e_y - dyc_deriv_state) / 0.005
            dyc_deriv_state += dde * dt
            Nz_classical = 4000.0 * e_y + 35.0 * dyc_integ + 1000.0 * dde + 5000.0 * e_phi
            
            # High-speed yaw damping
            v_blend = np.clip((Vx - 10.0) / 10.0, 0.0, 1.0)
            Nz_damp = -1500.0 * v_blend * gamma
            Nz_cmd = Nz_classical + delta_Nz_rl + Nz_damp
            
            # Directional ESC
            if abs(gamma) > 0.20 and (Nz_cmd * gamma > 0.0):
                scale = max(0.0, 1.0 - (abs(gamma) - 0.20) / 0.10)
                Nz_cmd *= scale
                
            Nz_filtered += (Nz_cmd - Nz_filtered) * (dt / 0.002)
            delta_T = Nz_filtered * (R / (2.0 * d))
            
            T_base_f = T_all * r_front * 0.5
            T_base_r = T_all * (1.0 - r_front) * 0.5
            T_cmd_raw = np.array([
                T_base_f + delta_T,
                T_base_f - delta_T,
                T_base_r + delta_T,
                T_base_r - delta_T
            ])
            
            # Cascade TCS
            wheel_speeds = w_wheels * R
            delta_Vw = wheel_speeds - Vx
            delta_V_allow = np.maximum(0.12, 0.07 * Vx)
            excess_slip = np.maximum(0.0, delta_Vw - delta_V_allow)
            T_cmds = np.maximum(0.0, T_cmd_raw - 900.0 * excess_slip)
            T_FR_cmd, T_FL_cmd, T_RR_cmd, T_RL_cmd = T_cmds[0], T_cmds[1], T_cmds[2], T_cmds[3]
            
            # Morimoto Id*
            we_f = sim.pn_f * max(0.0, Vx / R)
            num_f = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f * (we_f**2)
            den_f = sim.Ra_f * (sim.Rce_f**2) + (sim.Ra_f + sim.Rce_f) * ((sim.Ld_f * we_f)**2)
            id_f = num_f / den_f if den_f > 0 else 0.0
            
            we_r = sim.pn_r * max(0.0, Vx / R)
            num_r = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r * (we_r**2)
            den_r = sim.Ra_r * (sim.Rce_r**2) + (sim.Ra_r + sim.Rce_r) * ((sim.Ld_r * we_r)**2)
            id_r = num_r / den_r if den_r > 0 else 0.0
            Id_FR, Id_FL, Id_RR, Id_RL = id_f, id_f, id_r, id_r
            
        # EPS dynamics
        delta_f_cmd = np.clip(delta_f_cmd, -sim.EPS_Del_max, sim.EPS_Del_max)
        delta_f_actual += (delta_f_cmd - delta_f_actual) * (dt / sim.EPS_tau)
        
        # Motor dynamics
        T_cmds = np.array([T_FR_cmd, T_FL_cmd, T_RR_cmd, T_RL_cmd])
        T_maxes = np.array([sim.Tmax_f, sim.Tmax_f, sim.Tmax_r, sim.Tmax_r])
        T_clipped = np.clip(T_cmds, -T_maxes, T_maxes)
        T_actual += (T_clipped - T_actual) * (dt / sim.Motor_tau)
        
        # Pacejka Tires
        beta = np.arctan2(Vy, Vx) if abs(Vx) > 0.1 else 0.0
        alpha_fr = delta_f_actual - beta - (lf * gamma) / max(0.1, Vx)
        alpha_fl = delta_f_actual - beta - (lf * gamma) / max(0.1, Vx)
        alpha_rr = -beta + (lr * gamma) / max(0.1, Vx)
        alpha_rl = -beta + (lr * gamma) / max(0.1, Vx)
        
        Fx_fr, Fy_fr, lam_fr = sim.pacejka(Vx, w_wheels[0]*R, alpha_fr, mu, Nfr)
        Fx_fl, Fy_fl, lam_fl = sim.pacejka(Vx, w_wheels[1]*R, alpha_fl, mu, Nfl)
        Fx_rr, Fy_rr, lam_rr = sim.pacejka(Vx, w_wheels[2]*R, alpha_rr, mu, Nrr)
        Fx_rl, Fy_rl, lam_rl = sim.pacejka(Vx, w_wheels[3]*R, alpha_rl, mu, Nrl)
        
        # Wheel rotational dynamics
        w_wheels[0] += ((T_actual[0] - R * Fx_fr) / sim.Jw_f) * dt
        w_wheels[1] += ((T_actual[1] - R * Fx_fl) / sim.Jw_f) * dt
        w_wheels[2] += ((T_actual[2] - R * Fx_rr) / sim.Jw_r) * dt
        w_wheels[3] += ((T_actual[3] - R * Fx_rl) / sim.Jw_r) * dt
        
        # Vehicle Body Dynamics
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
        
        gamma_val = gamma
        if abs(gamma_val) > 0.20:
            gamma_val = np.clip(gamma_val, -0.20, 0.20)
            
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
        
        if abs(PosY) > max_y: max_y = abs(PosY)
        
        # Power & metrics
        Pin_fr, _ = sim.calc_motor_power(T_actual[0], w_wheels[0], True, Id_FR)
        Pin_fl, _ = sim.calc_motor_power(T_actual[1], w_wheels[1], True, Id_FL)
        Pin_rr, _ = sim.calc_motor_power(T_actual[2], w_wheels[2], False, Id_RR)
        Pin_rl, _ = sim.calc_motor_power(T_actual[3], w_wheels[3], False, Id_RL)
        P_all = Pin_fr + Pin_fl + Pin_rr + Pin_rl
        eta_dwpt = sim.get_dwpt_eta(PosX, PosY) / 100.0
        
        Phi1_sum += P_all * dt / 3600.0
        Phi2_sum += (lam_fl**2 + lam_fr**2 + lam_rl**2 + lam_rr**2) * dt
        Phi3_sum += (1.0 - eta_dwpt)**2 * dt
        Phi4_sum += ay**2 * dt
        
    total_Phi = 1.0 * Phi1_sum + 60.0 * Phi2_sum + 3.0 * Phi3_sum + 10.0 * Phi4_sum
    return {
        'Phi1': Phi1_sum,
        'Phi2': Phi2_sum,
        'Phi3': Phi3_sum,
        'Phi4': Phi4_sum,
        'total_Phi': total_Phi,
        'max_y': max_y
    }


def main():
    cond_names = [
        "Condition 1: High mu (0.85), No Wind",
        "Condition 2: High mu (0.85), Wind 1600N",
        "Condition 3: mu-change (0.25), No Wind",
        "Condition 4: mu-change (0.25), Wind 1600N"
    ]
    
    print("=" * 90)
    print("  OFFICIAL IEEE MVC 2027 BENCHMARK SCORING")
    print("  Formula: Phi = 1*Phi1 + 60*Phi2 + 3*Phi3 + 10*Phi4")
    print("=" * 90)
    
    results = {1: [], 2: [], 3: []}
    
    for c_id in [1, 2, 3, 4]:
        print(f"\n>>> Running {cond_names[c_id-1]}...")
        r_bs = run_single_simulation(1, c_id)
        r_is = run_single_simulation(2, c_id)
        r_prop = run_single_simulation(3, c_id)
        results[1].append(r_bs)
        results[2].append(r_is)
        results[3].append(r_prop)
        
        print("  [BS]   Baseline:  Phi=%7.2f (Phi1=%5.1f, 60*Phi2=%6.2f, 3*Phi3=%5.1f, 10*Phi4=%5.1f, max|y|=%5.1fmm)" % (
            r_bs['total_Phi'], r_bs['Phi1'], 60*r_bs['Phi2'], 3*r_bs['Phi3'], 10*r_bs['Phi4'], r_bs['max_y']*1000
        ))
        print("  [IS]   Improved:  Phi=%7.2f (Phi1=%5.1f, 60*Phi2=%6.2f, 3*Phi3=%5.1f, 10*Phi4=%5.1f, max|y|=%5.1fmm)" % (
            r_is['total_Phi'], r_is['Phi1'], 60*r_is['Phi2'], 3*r_is['Phi3'], 10*r_is['Phi4'], r_is['max_y']*1000
        ))
        print("  [OURS] Proposed:  Phi=%7.2f (Phi1=%5.1f, 60*Phi2=%6.2f, 3*Phi3=%5.1f, 10*Phi4=%5.1f, max|y|=%5.1fmm)" % (
            r_prop['total_Phi'], r_prop['Phi1'], 60*r_prop['Phi2'], 3*r_prop['Phi3'], 10*r_prop['Phi4'], r_prop['max_y']*1000
        ))
        
    print("\n" + "=" * 90)
    print("  FINAL SCORECARD: TOTAL PHI ACROSS ALL 4 CONDITIONS")
    print("=" * 90)
    total_bs = sum(r['total_Phi'] for r in results[1])
    total_is = sum(r['total_Phi'] for r in results[2])
    total_prop = sum(r['total_Phi'] for r in results[3])
    
    print(f"  1. Baseline Strategy (BS):  Total Phi = {total_bs:.2f}")
    print(f"  2. Improved Strategy (IS):  Total Phi = {total_is:.2f} ({(total_is - total_bs)/total_bs*100:+.1f}%)")
    print(f"  3. Proposed Strategy (Ours): Total Phi = {total_prop:.2f} ({(total_prop - total_bs)/total_bs*100:+.1f}% vs BS)")
    print("=" * 90)

if __name__ == '__main__':
    main()
