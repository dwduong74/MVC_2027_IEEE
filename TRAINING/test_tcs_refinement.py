import numpy as np
import time
from env_kanon_vehicle import KanonSimulator

def simulate_refined_strategy(flag_wind=1, flag_road=1, dt=0.0002, Tsim=120.0):
    sim = KanonSimulator()
    N_steps = int(Tsim / dt)
    
    # States
    Vx = 0.0
    Vy = 0.0
    gamma = 0.0
    PosX = 0.0
    PosY = 0.0
    phi = 0.0
    
    w_FR = 0.0
    w_FL = 0.0
    w_RR = 0.0
    w_RL = 0.0
    
    delta_f_actual = 0.0
    T_actual = np.zeros(4)
    
    v_integ = 0.0
    v_ff_state = 0.0
    
    dyc_integ = 0.0
    dyc_deriv_state = 0.0
    
    Phi1_Pin_sum = 0.0
    Phi2_lambda_sum = 0.0
    Phi3_dwpt_sum = 0.0
    Phi4_ay_sum = 0.0
    
    M, I = sim.M, sim.I
    lf, lr, d, hg = sim.lf, sim.lr, sim.d, sim.hg
    l = lf + lr
    R = sim.R
    Nf_static = sim.Nf
    Nr_static = sim.Nr
    
    ax_mem = 0.0
    ay_mem = 0.0
    
    print(f"Starting refined simulation: Wind: {flag_wind}, Road: {flag_road}...")
    t0 = time.time()
    
    # Store history for analysis
    slips_max = []
    
    for k in range(N_steps):
        t = k * dt
        
        # 1. References & Environment
        V_ref = sim.get_vref(t)
        mu = sim.get_road_friction(PosX, flag_road)
        F_wind, M_wind = sim.get_wind(t, flag_wind)
        
        # 2. Velocity Controller
        Delta_V = V_ref - Vx
        dv_ff = (V_ref - v_ff_state) / sim.tau_ff_v
        v_ff_state += dv_ff * dt
        F_ff = (M / sim.tau_ff_v) * (V_ref - v_ff_state)
        
        v_integ += Delta_V * dt
        F_fb = sim.Kp_v * Delta_V + sim.Ki_v * v_integ
        
        F_total = np.clip(F_ff + F_fb, -sim.Fmax, sim.Fmax)
        T_all = F_total * R
        
        # 3. Dynamic Normal Loads
        Nfr = Nf_static - (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lr / l)) * ay_mem
        Nfl = Nf_static - (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lr / l)) * ay_mem
        Nrr = Nr_static + (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lf / l)) * ay_mem
        Nrl = Nr_static + (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lf / l)) * ay_mem
        N_wheels = np.maximum(100.0, np.array([Nfr, Nfl, Nrr, Nrl]))
        
        # 4. REFINED HYBRID STRATEGY:
        # AFS steering:
        e_y = 0.0 - PosY
        e_phi = 0.0 - phi
        delta_f_cmd = 0.02 * e_y + 0.2 * e_phi
        
        # DYC Yaw Moment:
        dyc_integ += e_y * dt
        dde = (e_y - dyc_deriv_state) / 0.005
        dyc_deriv_state += dde * dt
        Nz_cmd = 4000.0 * e_y + 20.0 * dyc_integ + 1000.0 * dde + 5000.0 * e_phi
        delta_T = Nz_cmd * (R / (2.0 * d))
        
        # --- CASCADE TRACTION CONTROL (Ueno et al. 2025) ---
        # Calculate individual wheel slip speeds: Delta_Vw = w*R - Vx
        wheel_speeds = np.array([w_FR, w_FL, w_RR, w_RL]) * R
        delta_Vw = wheel_speeds - Vx
        
        # Calculate slip ratio for monitoring (using Pacejka definition with eps=0.05)
        # s_i = (w*R - Vx) / max(w*R, Vx, 0.05)
        # Allowable slip speed for maximum tractive force:
        # At Vx >= 1.0 m/s: Delta_V_allow = 0.08 * Vx (8% slip ratio)
        # At Vx < 1.0 m/s: Delta_V_allow = 0.15 m/s
        delta_V_allow = np.maximum(0.15, 0.08 * Vx)
        
        # Excess slip speed:
        excess_slip = np.maximum(0.0, delta_Vw - delta_V_allow)
        
        # Base torque allocation (equal 4 wheels or dynamic load based):
        T_base = T_all * 0.25
        
        T_cmd_raw = np.array([
            T_base + delta_T, # FR
            T_base - delta_T, # FL
            T_base + delta_T, # RR
            T_base - delta_T  # RL
        ])
        
        # TCS feedback: reduce torque proportionally to excess slip speed
        # K_tcs in [Nm / (m/s)]
        K_tcs = 800.0
        T_tcs_reduction = K_tcs * excess_slip
        
        T_cmds = np.maximum(0.0, T_cmd_raw - T_tcs_reduction)
        
        # --- LOSS-MINIMIZING D-AXIS CURRENT (Morimoto 1994) ---
        we_f = sim.pn_f * max(0.0, Vx / R)
        num_f = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f * (we_f**2)
        den_f = sim.Ra_f * (sim.Rce_f**2) + (sim.Ra_f + sim.Rce_f) * ((sim.Ld_f * we_f)**2)
        id_f = num_f / den_f if den_f > 0 else 0.0
        
        we_r = sim.pn_r * max(0.0, Vx / R)
        num_r = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r * (we_r**2)
        den_r = sim.Ra_r * (sim.Rce_r**2) + (sim.Ra_r + sim.Rce_r) * ((sim.Ld_r * we_r)**2)
        id_r = num_r / den_r if den_r > 0 else 0.0
        
        Id_FR, Id_FL = id_f, id_f
        Id_RR, Id_RL = id_r, id_r

        # 5. Actuators
        delta_f_cmd = np.clip(delta_f_cmd, -sim.EPS_Del_max, sim.EPS_Del_max)
        delta_f_actual += (delta_f_cmd - delta_f_actual) * (dt / sim.EPS_tau)
        
        T_limits = np.array([sim.Tmax_f, sim.Tmax_f, sim.Tmax_r, sim.Tmax_r])
        T_cmds = np.clip(T_cmds, -T_limits, T_limits)
        T_actual += (T_cmds - T_actual) * (dt / sim.Motor_tau)
        
        # 6. Tire Velocities and Slip Angles
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
        
        # 7. Tire Forces
        Fx_fr, Fy_fr, lam_fr = sim.pacejka(V_fr, w_FR * R, alpha_fr, mu, Nfr)
        Fx_fl, Fy_fl, lam_fl = sim.pacejka(V_fl, w_FL * R, alpha_fl, mu, Nfl)
        Fx_rr, Fy_rr, lam_rr = sim.pacejka(V_rr, w_RR * R, alpha_rr, mu, Nrr)
        Fx_rl, Fy_rl, lam_rl = sim.pacejka(V_rl, w_RL * R, alpha_rl, mu, Nrl)
        
        # 8. Wheel Dynamics
        w_FR += ((T_actual[0] - R * Fx_fr) / sim.Jw_f) * dt
        w_FL += ((T_actual[1] - R * Fx_fl) / sim.Jw_f) * dt
        w_RR += ((T_actual[2] - R * Fx_rr) / sim.Jw_r) * dt
        w_RL += ((T_actual[3] - R * Fx_rl) / sim.Jw_r) * dt
        
        # 9. Vehicle Dynamics
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
        ax_mem = ax
        ay_mem = ay
        
        dVx = ax + Vy * gamma_val
        dVy = ay - Vx * gamma_val
        dgamma = Nz_tot / I
        
        Vx += dVx * dt
        Vy += dVy * dt
        gamma += dgamma * dt
        
        dPosX = Vx * np.cos(phi) - Vy * np.sin(phi)
        dPosY = Vx * np.sin(phi) + Vy * np.cos(phi)
        dphi = gamma_val
        
        PosX += dPosX * dt
        PosY += dPosY * dt
        phi += dphi * dt
        
        # 10. Energy
        Pin_fr, _ = sim.calc_motor_power(T_actual[0], w_FR, True, Id_FR)
        Pin_fl, _ = sim.calc_motor_power(T_actual[1], w_FL, True, Id_FL)
        Pin_rr, _ = sim.calc_motor_power(T_actual[2], w_RR, False, Id_RR)
        Pin_rl, _ = sim.calc_motor_power(T_actual[3], w_RL, False, Id_RL)
        P_all = Pin_fr + Pin_fl + Pin_rr + Pin_rl
        
        # 11. DWPT
        eta_dwpt = sim.get_dwpt_eta(PosX, PosY) / 100.0
        
        Phi1_Pin_sum += P_all
        Phi2_lambda_sum += (lam_fl**2 + lam_fr**2 + lam_rl**2 + lam_rr**2)
        Phi3_dwpt_sum += (1.0 - eta_dwpt)**2
        Phi4_ay_sum += ay**2
        
    Phi1 = Phi1_Pin_sum * dt / 3600.0
    Phi2 = Phi2_lambda_sum * dt
    Phi3 = Phi3_dwpt_sum * dt
    Phi4 = Phi4_ay_sum * dt
    
    sigma = [1.0, 60.0, 3.0, 10.0]
    Phi = sigma[0]*Phi1 + sigma[1]*Phi2 + sigma[2]*Phi3 + sigma[3]*Phi4
    
    elapsed = time.time() - t0
    print(f"Refined simulation completed in {elapsed:.2f}s!")
    print(f"Phi1 (Energy Wh)         : {Phi1:.4f}")
    print(f"Phi2 (Slip penalty s)    : {Phi2:.4f} (Weighted: {sigma[1]*Phi2:.2f})")
    print(f"Phi3 (DWPT penalty s)    : {Phi3:.4f} (Weighted: {sigma[2]*Phi3:.2f})")
    print(f"Phi4 (Comfort m^2/s^3)   : {Phi4:.4f} (Weighted: {sigma[3]*Phi4:.2f})")
    print(f"TOTAL SCORE Phi          : {Phi:.4f}")
    print(f"Final PosX: {PosX:.2f}m, Max PosY: {PosY:.4f}m\n")
    return Phi

if __name__ == "__main__":
    simulate_refined_strategy(flag_wind=1, flag_road=1)
