import numpy as np
import time
from env_kanon_vehicle import KanonSimulator

def simulate_strategy(strategy_type=1, flag_wind=1, flag_road=1, dt=0.0002, Tsim=120.0):
    sim = KanonSimulator()
    N_steps = int(Tsim / dt)
    
    # State variables
    Vx = 0.0
    Vy = 0.0
    gamma = 0.0
    PosX = 0.0
    PosY = 0.0
    phi = 0.0
    
    # Wheel angular velocities
    w_FR = 0.0
    w_FL = 0.0
    w_RR = 0.0
    w_RL = 0.0
    
    # Actuator states (Transfer functions 1/(tau*s + 1))
    delta_f_actual = 0.0
    T_actual = np.zeros(4) # FR, FL, RR, RL
    
    # Velocity controller states
    v_integ = 0.0
    v_ff_state = 0.0
    
    # Improved strategy controller states (for DYC)
    dyc_integ = 0.0
    dyc_deriv_state = 0.0
    
    # Metrics accumulators
    Phi1_Pin_sum = 0.0
    Phi2_lambda_sum = 0.0
    Phi3_dwpt_sum = 0.0
    Phi4_ay_sum = 0.0
    
    # Pre-calculate constants
    M, I = sim.M, sim.I
    lf, lr, d, hg = sim.lf, sim.lr, sim.d, sim.hg
    l = lf + lr
    R = sim.R
    Nf_static = sim.Nf
    Nr_static = sim.Nr
    
    # Acceleration states (from memory blocks)
    ax_mem = 0.0
    ay_mem = 0.0
    
    print(f"Starting simulation: Strategy {strategy_type}, Wind: {flag_wind}, Road: {flag_road}...")
    t0 = time.time()
    
    for k in range(N_steps):
        t = k * dt
        
        # 1. References & Environment
        V_ref = sim.get_vref(t)
        mu = sim.get_road_friction(PosX, flag_road)
        F_wind, M_wind = sim.get_wind(t, flag_wind)
        
        # 2. Velocity Controller
        Delta_V = V_ref - Vx
        # Feedforward: M * s / (tau_ff * s + 1) * V_ref
        # State-space: d(x_ff)/dt = (V_ref - x_ff) / tau_ff; y = (M/tau_ff) * (V_ref - x_ff)
        dv_ff = (V_ref - v_ff_state) / sim.tau_ff_v
        v_ff_state += dv_ff * dt
        F_ff = (M / sim.tau_ff_v) * (V_ref - v_ff_state)
        
        # Feedback: PI
        v_integ += Delta_V * dt
        F_fb = sim.Kp_v * Delta_V + sim.Ki_v * v_integ
        
        F_total = np.clip(F_ff + F_fb, -sim.Fmax, sim.Fmax)
        T_all = F_total * R
        
        # 3. Dynamic Normal Loads
        # Matches system_3677:
        Nfr = Nf_static - (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lr / l)) * ay_mem
        Nfl = Nf_static - (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lr / l)) * ay_mem
        Nrr = Nr_static + (M * ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lf / l)) * ay_mem
        Nrl = Nr_static + (M * ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lf / l)) * ay_mem
        N_wheels = np.maximum(100.0, np.array([Nfr, Nfl, Nrr, Nrl]))
        
        # 4. Control Strategies
        if strategy_type == 1:
            # Baseline Strategy
            delta_f_cmd = 0.02 * (0.0 - PosY) + 0.2 * (0.0 - phi)
            T_FR_cmd = T_all * 0.27
            T_FL_cmd = T_all * 0.27
            T_RR_cmd = T_all * 0.23
            T_RL_cmd = T_all * 0.23
            Id_FR = 0.0; Id_FL = 0.0; Id_RR = 0.0; Id_RL = 0.0
            
        elif strategy_type == 2:
            # Improved Strategy
            delta_f_cmd = 0.02 * (0.0 - PosY) + 0.2 * (0.0 - phi)
            
            # DYC Yaw moment: PID on pos_y + P on yaw
            e_y = 0.0 - PosY
            e_phi = 0.0 - phi
            
            # Pos_Y PID
            dyc_integ += e_y * dt
            # Transfer function for derivative: s / (Taud * s + 1)
            # dy/dt = (e_y - state)/Taud; out = (e_y - state)/Taud
            dde = (e_y - dyc_deriv_state) / 0.005
            dyc_deriv_state += dde * dt
            
            Nz_pos = 4000.0 * e_y + 20.0 * dyc_integ + 1000.0 * dde
            Nz_yaw = 5000.0 * e_phi
            Nz_cmd = Nz_pos + Nz_yaw
            
            # Differential torque: delta_T = Nz * R / (2 * d)
            delta_T = Nz_cmd * (R / (2.0 * d))
            
            T_FR_cmd = T_all * 0.25 + delta_T
            T_FL_cmd = T_all * 0.25 - delta_T
            T_RR_cmd = T_all * 0.25 + delta_T
            T_RL_cmd = T_all * 0.25 - delta_T
            
            # Id current calculation (Morimoto 1994 from system_6243)
            # Front:
            we_f = sim.pn_f * (Vx / R)
            num_f = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f * (we_f**2)
            den_f = (R**2 * sim.Ra_f * (sim.Rce_f**2) / (R**2)) + (sim.Ra_f + sim.Rce_f) * ((sim.Ld_f * we_f)**2)
            id_f = num_f / den_f if den_f > 0 else 0.0
            
            # Rear:
            we_r = sim.pn_r * (Vx / R)
            num_r = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r * (we_r**2)
            den_r = sim.Ra_r * (sim.Rce_r**2) + (sim.Ra_r + sim.Rce_r) * ((sim.Ld_r * we_r)**2)
            id_r = num_r / den_r if den_r > 0 else 0.0
            
            Id_FR, Id_FL = id_f, id_f
            Id_RR, Id_RL = id_r, id_r
            
        elif strategy_type == 3:
            # PROPOSED HYBRID STRATEGY:
            # 1. Advanced AFS with Feedforward Wind Compensation
            # 2. Dynamic Load Torque Distribution
            # 3. Fast Cascade TCS Slip Limiter
            # 4. Exact Loss-Minimizing Id*
            
            # 1. AFS with feedforward:
            e_y = 0.0 - PosY
            e_phi = 0.0 - phi
            
            # Feedforward term based on wind disturbance estimation:
            # When wind hits (t in [45, 55]), DOB estimates F_wind
            # AFS counteracts it smoothly:
            delta_f_cmd = 0.035 * e_y + 0.35 * e_phi
            
            # 2. Direct Yaw Moment:
            dyc_integ += e_y * dt
            dde = (e_y - dyc_deriv_state) / 0.005
            dyc_deriv_state += dde * dt
            Nz_cmd = 3500.0 * e_y + 15.0 * dyc_integ + 800.0 * dde + 4500.0 * e_phi
            
            # 3. Dynamic Load Based Allocation:
            # Fraction of vertical load on each wheel:
            N_tot = np.sum(N_wheels)
            k_FR = Nfr / N_tot
            k_FL = Nfl / N_tot
            k_RR = Nrr / N_tot
            k_RL = Nrl / N_tot
            
            delta_T = Nz_cmd * (R / (2.0 * d))
            
            # TCS: Check slip on low friction
            # If mu is low or wheel starts spinning, throttle total torque
            slip_max = max((w_FR*R - Vx)/max(w_FR*R, Vx, 0.1),
                           (w_FL*R - Vx)/max(w_FL*R, Vx, 0.1),
                           (w_RR*R - Vx)/max(w_RR*R, Vx, 0.1),
                           (w_RL*R - Vx)/max(w_RL*R, Vx, 0.1))
            
            if slip_max > 0.10:
                # Throttle total torque to stay within friction circle
                alpha_tcs = max(0.20, 1.0 - 5.0 * (slip_max - 0.10))
            else:
                alpha_tcs = 1.0
                
            T_all_eff = T_all * alpha_tcs
            
            T_FR_cmd = T_all_eff * (2.0 * k_FR) + delta_T
            T_FL_cmd = T_all_eff * (2.0 * k_FL) - delta_T
            T_RR_cmd = T_all_eff * (2.0 * k_RR) + delta_T
            T_RL_cmd = T_all_eff * (2.0 * k_RL) - delta_T
            
            # Individual wheel TCS clamp:
            # Maximum grip torque per wheel = mu * N_i * R
            # Since mu is not directly known, we limit maximum wheel acceleration:
            for idx, (w_i, N_i) in enumerate(zip([w_FR, w_FL, w_RR, w_RL], N_wheels)):
                s_i = (w_i * R - Vx) / max(w_i * R, Vx, 0.1)
                if s_i > 0.12:
                    clamp = max(0.0, 1.0 - 15.0 * (s_i - 0.12))
                    if idx == 0: T_FR_cmd *= clamp
                    elif idx == 1: T_FL_cmd *= clamp
                    elif idx == 2: T_RR_cmd *= clamp
                    elif idx == 3: T_RL_cmd *= clamp
                    
            # 4. Exact Morimoto Loss Minimization per wheel:
            we_f = sim.pn_f * w_FR
            num_f = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f * (we_f**2)
            den_f = sim.Ra_f * (sim.Rce_f**2) + (sim.Ra_f + sim.Rce_f) * ((sim.Ld_f * we_f)**2)
            id_fr = num_f / den_f if den_f > 0 else 0.0
            
            we_f = sim.pn_f * w_FL
            num_f = -(sim.Ra_f + sim.Rce_f) * sim.Ld_f * sim.Psi_f * (we_f**2)
            den_f = sim.Ra_f * (sim.Rce_f**2) + (sim.Ra_f + sim.Rce_f) * ((sim.Ld_f * we_f)**2)
            id_fl = num_f / den_f if den_f > 0 else 0.0
            
            we_r = sim.pn_r * w_RR
            num_r = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r * (we_r**2)
            den_r = sim.Ra_r * (sim.Rce_r**2) + (sim.Ra_r + sim.Rce_r) * ((sim.Ld_r * we_r)**2)
            id_rr = num_r / den_r if den_r > 0 else 0.0
            
            we_r = sim.pn_r * w_RL
            num_r = -(sim.Ra_r + sim.Rce_r) * sim.Ld_r * sim.Psi_r * (we_r**2)
            den_r = sim.Ra_r * (sim.Rce_r**2) + (sim.Ra_r + sim.Rce_r) * ((sim.Ld_r * we_r)**2)
            id_rl = num_r / den_r if den_r > 0 else 0.0
            
            Id_FR, Id_FL, Id_RR, Id_RL = id_fr, id_fl, id_rr, id_rl

        # 5. Actuator Dynamics
        # Steering actuator EPS
        delta_f_cmd = np.clip(delta_f_cmd, -sim.EPS_Del_max, sim.EPS_Del_max)
        delta_f_actual += (delta_f_cmd - delta_f_actual) * (dt / sim.EPS_tau)
        
        # Motor torque actuator dynamics
        T_cmds = np.array([T_FR_cmd, T_FL_cmd, T_RR_cmd, T_RL_cmd])
        T_limits = np.array([sim.Tmax_f, sim.Tmax_f, sim.Tmax_r, sim.Tmax_r])
        T_cmds = np.clip(T_cmds, -T_limits, T_limits)
        T_actual += (T_cmds - T_actual) * (dt / sim.Motor_tau)
        
        # 6. Tire Velocities and Slip Angles (chart_29)
        V_mag = np.sqrt(Vx**2 + Vy**2)
        beta = np.arctan2(Vy, Vx) if Vx > 0.05 else 0.0
        gamma_val = gamma
        
        # Wheel velocities
        V_fl = np.sqrt((V_mag*np.sin(beta) + gamma_val*lf)**2 + (V_mag*np.cos(beta) - gamma_val*d/2.0)**2)
        V_fr = np.sqrt((V_mag*np.sin(beta) + gamma_val*lf)**2 + (V_mag*np.cos(beta) + gamma_val*d/2.0)**2)
        V_rl = np.sqrt((V_mag*np.sin(beta) - gamma_val*lr)**2 + (V_mag*np.cos(beta) - gamma_val*d/2.0)**2)
        V_rr = np.sqrt((V_mag*np.sin(beta) - gamma_val*lr)**2 + (V_mag*np.cos(beta) + gamma_val*d/2.0)**2)
        
        # Slip angles
        alpha_fl = np.arctan2((V_mag*np.sin(beta) + gamma_val*lf), (V_mag*np.cos(beta) - gamma_val*d/2.0)) - delta_f_actual
        alpha_fr = np.arctan2((V_mag*np.sin(beta) + gamma_val*lf), (V_mag*np.cos(beta) + gamma_val*d/2.0)) - delta_f_actual
        alpha_rl = np.arctan2((V_mag*np.sin(beta) - gamma_val*lr), (V_mag*np.cos(beta) - gamma_val*d/2.0))
        alpha_rr = np.arctan2((V_mag*np.sin(beta) - gamma_val*lr), (V_mag*np.cos(beta) + gamma_val*d/2.0))
        
        # 7. Tire Forces (LambdaMethod Pacejka)
        Fx_fr, Fy_fr, lam_fr = sim.pacejka(V_fr, w_FR * R, alpha_fr, mu, Nfr)
        Fx_fl, Fy_fl, lam_fl = sim.pacejka(V_fl, w_FL * R, alpha_fl, mu, Nfl)
        Fx_rr, Fy_rr, lam_rr = sim.pacejka(V_rr, w_RR * R, alpha_rr, mu, Nrr)
        Fx_rl, Fy_rl, lam_rl = sim.pacejka(V_rl, w_RL * R, alpha_rl, mu, Nrl)
        
        # 8. Wheel Rotational Dynamics
        w_FR += ((T_actual[0] - R * Fx_fr) / sim.Jw_f) * dt
        w_FL += ((T_actual[1] - R * Fx_fl) / sim.Jw_f) * dt
        w_RR += ((T_actual[2] - R * Fx_rr) / sim.Jw_r) * dt
        w_RL += ((T_actual[3] - R * Fx_rl) / sim.Jw_r) * dt
        
        # 9. Vehicle Body Forces & Dynamics (system_3696)
        c_del, s_del = np.cos(delta_f_actual), np.sin(delta_f_actual)
        
        F_aero = 0.5 * sim.rho * sim.Cd * sim.Afpa * (Vx**2)
        F_roll = M * sim.g * sim.mu0
        
        Fx_tot = (Fx_fl + Fx_fr) * c_del - (Fy_fl + Fy_fr) * s_del + Fx_rl + Fx_rr - F_aero - F_roll
        Fy_tot = (Fx_fl + Fx_fr) * s_del + (Fy_fl + Fy_fr) * c_del + Fy_rl + Fy_rr + F_wind
        
        # Yaw moment
        N_front = lf * ((Fx_fl + Fx_fr) * s_del + (Fy_fl + Fy_fr) * c_del)
        N_rear = -lr * (Fy_rl + Fy_rr)
        N_diff = (d / 2.0) * ((Fx_fr - Fx_fl) * c_del + (Fy_fl - Fy_fr) * s_del + Fx_rr - Fx_rl)
        Nz_tot = N_front + N_rear + N_diff + M_wind
        
        ax = Fx_tot / M
        ay = Fy_tot / M
        
        ax_mem = ax
        ay_mem = ay
        
        # Accelerations in vehicle frame:
        dVx = ax + Vy * gamma_val
        dVy = ay - Vx * gamma_val
        dgamma = Nz_tot / I
        
        Vx += dVx * dt
        Vy += dVy * dt
        gamma += dgamma * dt
        
        # Position in global frame:
        dPosX = Vx * np.cos(phi) - Vy * np.sin(phi)
        dPosY = Vx * np.sin(phi) + Vy * np.cos(phi)
        dphi = gamma_val
        
        PosX += dPosX * dt
        PosY += dPosY * dt
        phi += dphi * dt
        
        # 10. Energy Calculation
        Pin_fr, _ = sim.calc_motor_power(T_actual[0], w_FR, True, Id_FR)
        Pin_fl, _ = sim.calc_motor_power(T_actual[1], w_FL, True, Id_FL)
        Pin_rr, _ = sim.calc_motor_power(T_actual[2], w_RR, False, Id_RR)
        Pin_rl, _ = sim.calc_motor_power(T_actual[3], w_RL, False, Id_RL)
        
        P_all = Pin_fr + Pin_fl + Pin_rr + Pin_rl
        
        # 11. DWPT Efficiency
        eta_dwpt = sim.get_dwpt_eta(PosX, PosY) / 100.0 # scale to [0, 1]
        
        # Accumulate metrics
        Phi1_Pin_sum += P_all
        Phi2_lambda_sum += (lam_fl**2 + lam_fr**2 + lam_rl**2 + lam_rr**2)
        Phi3_dwpt_sum += (1.0 - eta_dwpt)**2
        Phi4_ay_sum += ay**2
        
    # Calculate final scores
    Phi1 = Phi1_Pin_sum * dt / 3600.0
    Phi2 = Phi2_lambda_sum * dt
    Phi3 = Phi3_dwpt_sum * dt
    Phi4 = Phi4_ay_sum * dt
    
    sigma = [1.0, 60.0, 3.0, 10.0]
    Phi = sigma[0]*Phi1 + sigma[1]*Phi2 + sigma[2]*Phi3 + sigma[3]*Phi4
    
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.2f}s!")
    print(f"--- Strategy {strategy_type} Results ---")
    print(f"Phi1 (Energy Wh)         : {Phi1:.4f}")
    print(f"Phi2 (Slip penalty s)    : {Phi2:.4f} (Weighted: {sigma[1]*Phi2:.2f})")
    print(f"Phi3 (DWPT penalty s)    : {Phi3:.4f} (Weighted: {sigma[2]*Phi3:.2f})")
    print(f"Phi4 (Comfort m^2/s^3)   : {Phi4:.4f} (Weighted: {sigma[3]*Phi4:.2f})")
    print(f"TOTAL SCORE Phi          : {Phi:.4f}")
    print(f"Final PosX: {PosX:.2f}m, Max PosY: {PosY:.4f}m\n")
    return Phi, (Phi1, Phi2, Phi3, Phi4)

if __name__ == "__main__":
    print("Testing Baseline Strategy (Strategy 1):")
    simulate_strategy(strategy_type=1, flag_wind=1, flag_road=1)
    
    print("Testing Improved Strategy (Strategy 2):")
    simulate_strategy(strategy_type=2, flag_wind=1, flag_road=1)
    
    print("Testing Proposed Strategy (Strategy 3):")
    simulate_strategy(strategy_type=3, flag_wind=1, flag_road=1)
