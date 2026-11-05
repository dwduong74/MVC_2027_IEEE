import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from env_kanon_vehicle import KanonSimulator
from adversarial_stress_tester import build_curated_adversarial_suite

# Pre-load policy weights for fast numpy inference (no torch overhead)
_weights_file = os.path.join(os.path.dirname(__file__), "rl_policy_weights.json")
with open(_weights_file, 'r') as f:
    _raw_w = json.load(f)
_W1 = np.array(_raw_w['W1'], dtype=np.float64)
_b1 = np.array(_raw_w['b1'], dtype=np.float64)
_W2 = np.array(_raw_w['W2'], dtype=np.float64)
_b2 = np.array(_raw_w['b2'], dtype=np.float64)
_W_mu = np.array(_raw_w['W_mu'], dtype=np.float64)
_b_mu = np.array(_raw_w['b_mu'], dtype=np.float64)

def numpy_policy_infer(obs):
    h1 = np.tanh(_W1 @ obs + _b1)
    h2 = np.tanh(_W2 @ h1 + _b2)
    a = np.tanh(_W_mu @ h2 + _b_mu)
    return a

def simulate_with_controller(scenario, model_path=None,
                             lat_derate_slope=15.0, # scales down Tall as |PosY| grows
                             slip_derate_gain=20.0,  # scales down Tall when wheels slip
                             dyc_max_dt=250.0,      # limit max differential torque on wheels
                             yaw_damping=2500.0,    # high-speed yaw damping [Nm / (rad/s)]
                             tau_dyc=0.015):        # low-pass filter on DYC moment
    
    sim = KanonSimulator()
    if scenario.mass_scale != 1.0:
        sim.M *= scenario.mass_scale
        sim.I *= scenario.mass_scale
        sim.Nf *= scenario.mass_scale
        sim.Nr *= scenario.mass_scale
        
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
        
    orig_dwpt = sim.get_dwpt_eta
    def custom_dwpt_eta(x, y):
        if x < scenario.dwpt_x_start or x > scenario.dwpt_x_end: return 0.0
        return orig_dwpt(x, y)
        
    sim.get_road_friction = custom_road_friction
    sim.get_wind = custom_wind
    sim.get_vref = custom_vref
    sim.get_dwpt_eta = custom_dwpt_eta
    
    dt = 0.001
    dt_macro = 0.02
    substeps = 20
    Tsim = scenario.Tsim
    
    # State
    Vx, Vy, gamma = 0.0, 0.0, 0.0
    PosX, PosY, phi = 0.0, 0.0, 0.0
    w_wheels = np.zeros(4)
    delta_f_actual = 0.0
    T_actual = np.zeros(4)
    
    v_integ, v_ff_state = 0.0, 0.0
    dyc_integ, dyc_deriv_state = 0.0, 0.0
    Nz_filtered = 0.0
    ax_mem, ay_mem = 0.0, 0.0
    
    cum_Phi1, cum_Phi2, cum_Phi3, cum_Phi4 = 0.0, 0.0, 0.0, 0.0
    
    t = 0.0
    is_dq = False
    max_y = 0.0
    max_gamma = 0.0
    max_slip = 0.0
    
    R = sim.R
    M, I, lf, lr, d, hg = sim.M, sim.I, sim.lf, sim.lr, sim.d, sim.hg
    l = lf + lr
    Nf_static, Nr_static = sim.Nf, sim.Nr
    
    we_f_const = sim.pn_f / R
    we_r_const = sim.pn_r / R
    num_f_k = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f
    den_f_a = sim.Ra_f * (sim.Rce_f**2)
    den_f_b = (sim.Ra_f + sim.Rce_f) * (sim.Ld_f**2)
    num_r_k = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r
    den_r_a = sim.Ra_r * (sim.Rce_r**2)
    den_r_b = (sim.Ra_r + sim.Rce_r) * (sim.Ld_r**2)
    
    K_tcs = 900.0
    slip_ratio_allow = 0.07
    slip_speed_min = 0.12
    dyc_kp_pos = 4000.0
    dyc_ki_pos = 35.0
    dyc_kd_pos = 1000.0
    dyc_kp_yaw = 5000.0
    esc_gamma_limit = 0.20
    steer_kp_pos = 0.02
    steer_kp_yaw = 0.20
    
    N_macro = int(Tsim / dt_macro)
    
    for _ in range(N_macro):
        if t >= Tsim or is_dq:
            break
            
        # RL observation
        V_ref_m = sim.get_vref(t)
        v_err = V_ref_m - Vx
        speeds_m = w_wheels * R
        excess_m = np.max(speeds_m - Vx) if Vx > 0.1 else 0.0
        in_dwpt = 1.0 if (500.0 <= PosX <= 1500.0) else 0.0
        
        obs = np.array([
            Vx / 20.0,
            v_err / 5.0,
            np.clip(PosY / 0.10, -5.0, 5.0),
            np.clip(Vy / 2.0, -5.0, 5.0),
            np.clip(phi / 0.10, -5.0, 5.0),
            np.clip(gamma / 0.30, -5.0, 5.0),
            np.clip(ay_mem / 2.0, -5.0, 5.0),
            np.clip(excess_m / 0.15, 0.0, 5.0),
            in_dwpt,
            PosX / 2022.0
        ], dtype=np.float32)
        
        action = numpy_policy_infer(obs)
        a = np.clip(action, -1.0, 1.0)
        alpha_throttle = 1.0 + 0.20 * a[0]
        r_front = 0.50 + 0.15 * a[1]
        delta_Nz_rl = 500.0 * a[2]
        r_rear = 1.0 - r_front
        
        for _ in range(substeps):
            if t >= Tsim:
                break
                
            V_ref = sim.get_vref(t)
            mu = sim.get_road_friction(PosX, scenario.flag_road)
            F_wind, M_wind = sim.get_wind(t, scenario.flag_wind)
            
            # 1. Velocity Controller
            Delta_V = V_ref - Vx
            dv_ff = (V_ref - v_ff_state) / sim.tau_ff_v
            v_ff_state += dv_ff * dt
            F_ff = (M / sim.tau_ff_v) * (V_ref - v_ff_state)
            v_integ += Delta_V * dt
            F_fb = sim.Kp_v * Delta_V + sim.Ki_v * v_integ
            F_total = np.clip(F_ff + F_fb, -sim.Fmax, sim.Fmax)
            
            T_all = F_total * R * alpha_throttle
            
            # --- LATERAL-PRIORITY TRACTION DE-RATING ---
            # Monitor lateral tracking error and wheel slip
            wheel_speeds = w_wheels * R
            current_excess = np.max(wheel_speeds - Vx) if Vx > 0.1 else 0.0
            
            if T_all > 0:
                # Lateral demand penalty
                # e_lat = |PosY| / 0.08
                lat_metric = abs(PosY) / 0.08 + 0.5 * abs(phi) / 0.05
                # When slip is occurring or road friction is low, lateral priority is critical
                slip_metric = current_excess / 0.12
                
                # De-rate factor: reduces drive force when lateral error or slip occurs
                derate = 1.0 - np.clip(lat_derate_slope * (abs(PosY) - 0.015) + slip_derate_gain * slip_metric * abs(PosY), 0.0, 1.0)
                T_all *= max(0.0, derate)
                
            # Dynamic normal load
            Nfr = Nf_static - (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lr / l)) * ay_mem
            Nfl = Nf_static - (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lr / l)) * ay_mem
            Nrr = Nr_static + (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lf / l)) * ay_mem
            Nrl = Nr_static + (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lf / l)) * ay_mem
            
            # 3. Lateral & Yaw Control (AFS + DYC + High-Speed Yaw Damping)
            e_y = 0.0 - PosY
            e_phi = 0.0 - phi
            
            delta_f_cmd = steer_kp_pos * e_y + steer_kp_yaw * e_phi
            dyc_integ += e_y * dt
            dde = (e_y - dyc_deriv_state) / 0.005
            dyc_deriv_state += dde * dt
            
            Nz_classical = (dyc_kp_pos * e_y + 
                            dyc_ki_pos * dyc_integ + 
                            dyc_kd_pos * dde + 
                            dyc_kp_yaw * e_phi)
                            
            # High-speed yaw damping
            v_blend = np.clip((Vx - 10.0) / 10.0, 0.0, 1.0)
            Nz_damp = -yaw_damping * v_blend * gamma
            
            Nz_cmd = Nz_classical + delta_Nz_rl + Nz_damp
            
            # Smart ESC: only clamp if command is destabilizing
            if abs(gamma) > esc_gamma_limit and (Nz_cmd * gamma > 0.0):
                scale = max(0.0, 1.0 - (abs(gamma) - esc_gamma_limit) / 0.10)
                Nz_cmd *= scale
                
            # Slew rate / low-pass filter
            if tau_dyc > 0:
                Nz_filtered += (Nz_cmd - Nz_filtered) * (dt / tau_dyc)
                Nz_total = Nz_filtered
            else:
                Nz_total = Nz_cmd
                
            delta_T = Nz_total * (R / (2.0 * d))
            
            # Clamp delta_T on low friction / high slip so DYC doesn't overpower tires
            # On low mu, tire cannot take > 150 Nm
            if dyc_max_dt > 0:
                delta_T = np.clip(delta_T, -dyc_max_dt, dyc_max_dt)
                
            # 4. Torque distribution
            T_base_f = T_all * r_front * 0.5
            T_base_r = T_all * r_rear * 0.5
            
            T_cmd_raw = np.array([
                T_base_f + delta_T,
                T_base_f - delta_T,
                T_base_r + delta_T,
                T_base_r - delta_T
            ])
            
            # 5. Lateral-Priority TCS: reduce allowable slip when lateral error is present
            lat_priority = np.clip(1.0 - abs(PosY) / 0.08, 0.1, 1.0)
            effective_allow_slip = np.maximum(slip_speed_min * lat_priority, slip_ratio_allow * Vx * lat_priority)
            excess_slip = np.maximum(0.0, (w_wheels * R - Vx) - effective_allow_slip)
            
            T_cmds = np.maximum(0.0, T_cmd_raw - K_tcs * excess_slip)
            
            # 6. Loss-Minimization Id* (Morimoto)
            we_f = we_f_const * max(0.0, Vx)
            we_f_sq = we_f * we_f
            id_f = (num_f_k * we_f_sq) / (den_f_a + den_f_b * we_f_sq)
            
            we_r = we_r_const * max(0.0, Vx)
            we_r_sq = we_r * we_r
            id_r = (num_r_k * we_r_sq) / (den_r_a + den_r_b * we_r_sq)
            
            # 7. Actuators
            delta_f_cmd = np.clip(delta_f_cmd, -sim.EPS_Del_max, sim.EPS_Del_max)
            delta_f_actual += (delta_f_cmd - delta_f_actual) * (dt / sim.EPS_tau)
            T_limits = np.array([sim.Tmax_f, sim.Tmax_f, sim.Tmax_r, sim.Tmax_r])
            T_cmds = np.clip(T_cmds, -T_limits, T_limits)
            T_actual += (T_cmds - T_actual) * (dt / sim.Motor_tau)
            
            # 8. Tire velocities & Pacejka
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
            
            Fx_fr, Fy_fr, lam_fr = sim.pacejka(V_fr, w_wheels[0] * R, alpha_fr, mu, Nfr)
            Fx_fl, Fy_fl, lam_fl = sim.pacejka(V_fl, w_wheels[1] * R, alpha_fl, mu, Nfl)
            Fx_rr, Fy_rr, lam_rr = sim.pacejka(V_rr, w_wheels[2] * R, alpha_rr, mu, Nrr)
            Fx_rl, Fy_rl, lam_rl = sim.pacejka(V_rl, w_wheels[3] * R, alpha_rl, mu, Nrl)
            
            w_wheels[0] += ((T_actual[0] - R * Fx_fr) / sim.Jw_f) * dt
            w_wheels[1] += ((T_actual[1] - R * Fx_fl) / sim.Jw_f) * dt
            w_wheels[2] += ((T_actual[2] - R * Fx_rr) / sim.Jw_r) * dt
            w_wheels[3] += ((T_actual[3] - R * Fx_rl) / sim.Jw_r) * dt
            
            # 9. Vehicle Body Dynamics
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
            
            if abs(PosY) > max_y: max_y = abs(PosY)
            if abs(gamma) > max_gamma: max_gamma = abs(gamma)
            excess = np.max(w_wheels * R - Vx) if Vx > 0.1 else 0.0
            if excess > max_slip: max_slip = excess
            
            if abs(PosY) > 5.0:
                is_dq = True
                break
                
            Pin_fr, _ = sim.calc_motor_power(T_actual[0], w_wheels[0], True, id_f)
            Pin_fl, _ = sim.calc_motor_power(T_actual[1], w_wheels[1], True, id_f)
            Pin_rr, _ = sim.calc_motor_power(T_actual[2], w_wheels[2], False, id_r)
            Pin_rl, _ = sim.calc_motor_power(T_actual[3], w_wheels[3], False, id_r)
            P_all = Pin_fr + Pin_fl + Pin_rr + Pin_rl
            eta_dwpt = sim.get_dwpt_eta(PosX, PosY) / 100.0
            
            cum_Phi1 += P_all * dt / 3600.0
            cum_Phi2 += (lam_fl**2 + lam_fr**2 + lam_rl**2 + lam_rr**2) * dt
            cum_Phi3 += (1.0 - eta_dwpt)**2 * dt
            cum_Phi4 += ay**2 * dt
            
            t += dt
            
    total_score = 1.0 * cum_Phi1 + 60.0 * cum_Phi2 + 3.0 * cum_Phi3 + 10.0 * cum_Phi4
    return {
        'name': scenario.name,
        'is_dq': is_dq,
        'score': total_score,
        'p1': cum_Phi1,
        'p2': cum_Phi2,
        'p3': cum_Phi3,
        'p4': cum_Phi4,
        'max_y': max_y,
        'max_gamma': max_gamma,
        'max_slip': max_slip
    }

if __name__ == '__main__':
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, "latest_hybrid_ppo.pt")
    suite = build_curated_adversarial_suite()
    
    print("Testing ADV-01 and ADV-02 with Lateral-Priority TCS & Clamped DYC:")
    for s in suite[:2]:
        r = simulate_with_controller(s, model_path)
        status = "PASSED" if not r['is_dq'] else "DISQUALIFIED"
        print(f"[{status}] {r['name']}: Phi={r['score']:.2f}, max|y|={r['max_y']*1000:.1f}mm, 10*Phi4={10*r['p4']:.1f}, slip={r['max_slip']:.3f}m/s")
