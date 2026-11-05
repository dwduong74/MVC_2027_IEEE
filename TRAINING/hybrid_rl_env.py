"""
Hybrid RL Environment for IEEE MVC 2027
========================================
Combines:
  1. Macro-step RL Policy (50 Hz, dt = 0.02s):
     Outputs: [a_throttle, a_bias, a_yaw] in [-1, 1]
       - a_throttle: modulates total torque request alpha in [0.8, 1.2]
       - a_bias: front/rear torque bias ratio r_front in [0.35, 0.65]
       - a_yaw: auxiliary yaw moment delta_Nz in [-500, 500] Nm
  2. Micro-step Classical Controller (1000 Hz, dt = 0.001s):
     Executes:
       - Active Front Steering (AFS)
       - Direct Yaw Control (DYC)
       - Cascade Traction Control System (TCS)
       - ESC Yaw Rate Limiter
       - Morimoto Loss-Minimization Controller (Id*)
       - Full Pacejka 4-wheel dynamic load vehicle physics
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from env_kanon_vehicle import KanonSimulator


class HybridKanonEnv:
    def __init__(self, dt_macro=0.02, dt_micro=0.001):
        self.dt_macro = dt_macro
        self.dt_micro = dt_micro
        self.substeps = int(round(dt_macro / dt_micro))
        
        self.sim = KanonSimulator()
        self.Tsim = 120.0
        
        # Action space: 3 continuous dims in [-1, 1]
        self.action_dim = 3
        # State space: 10 continuous dims
        self.state_dim = 10
        
        # Precomputed motor constants
        self.R = self.sim.R
        self.we_f_const = self.sim.pn_f / self.R
        self.we_r_const = self.sim.pn_r / self.R
        self.num_f_k = -(self.sim.Ra_f + self.sim.Rce_f) * self.sim.Ld_f * self.sim.Psi_f
        self.den_f_a = self.sim.Ra_f * (self.sim.Rce_f**2)
        self.den_f_b = (self.sim.Ra_f + self.sim.Rce_f) * (self.sim.Ld_f**2)
        
        self.num_r_k = -(self.sim.Ra_r + self.sim.Rce_r) * self.sim.Ld_r * self.sim.Psi_r
        self.den_r_a = self.sim.Ra_r * (self.sim.Rce_r**2)
        self.den_r_b = (self.sim.Ra_r + self.sim.Rce_r) * (self.sim.Ld_r**2)
        
        # Tuned classical parameters (Gold Standard)
        self.K_tcs = 900.0
        self.slip_ratio_allow = 0.07
        self.slip_speed_min = 0.12
        self.dyc_kp_pos = 4000.0
        self.dyc_ki_pos = 35.0
        self.dyc_kd_pos = 1000.0
        self.dyc_kp_yaw = 5000.0
        self.esc_gamma_limit = 0.20
        self.steer_kp_pos = 0.02
        self.steer_kp_yaw = 0.20
        
        # High-Speed Yaw Damping and Lateral-Priority Limits
        self.yaw_damping = 2500.0       # High-speed yaw damping gain [Nm/(rad/s)]
        self.dyc_max_dt = 250.0         # Maximum differential torque clamp [Nm] per wheel
        self.tau_dyc = 0.015            # Low-pass filter time constant for DYC moment [s]
        self.lat_derate_slope = 15.0    # Lateral traction de-rating slope [1/m]
        self.slip_derate_gain = 20.0    # Slip-coupled traction de-rating gain [1/m]
        
        self.reset()
        
    def reset(self, condition_id=None):
        """
        condition_id in [1, 2, 3, 4] or None (random choice)
        """
        if condition_id is None:
            self.condition_id = np.random.choice([1, 2, 3, 4])
        else:
            self.condition_id = condition_id
            
        self.flag_wind = 1 if self.condition_id in [2, 4] else 0
        self.flag_road = 1 if self.condition_id in [3, 4] else 0
        
        self.t = 0.0
        self.step_count = 0
        
        # Vehicle states
        self.Vx = 0.0
        self.Vy = 0.0
        self.gamma = 0.0
        self.PosX = 0.0
        self.PosY = 0.0
        self.phi = 0.0
        self.w_wheels = np.zeros(4)
        
        # Actuators
        self.delta_f_actual = 0.0
        self.T_actual = np.zeros(4)
        
        # Controller internal states
        self.v_integ = 0.0
        self.v_ff_state = 0.0
        self.dyc_integ = 0.0
        self.dyc_deriv_state = 0.0
        self.Nz_filtered = 0.0
        
        # Running cost accumulators
        self.cum_Phi1 = 0.0
        self.cum_Phi2 = 0.0
        self.cum_Phi3 = 0.0
        self.cum_Phi4 = 0.0
        
        self.ax_mem = 0.0
        self.ay_mem = 0.0
        
        return self._get_obs()
        
    def _get_obs(self):
        V_ref = self.sim.get_vref(self.t)
        v_err = V_ref - self.Vx
        
        # Wheel max slip
        speeds = self.w_wheels * self.R
        delta_Vw = speeds - self.Vx
        max_excess = np.max(delta_Vw) if self.Vx > 0.1 else 0.0
        
        in_dwpt = 1.0 if (500.0 <= self.PosX <= 1500.0) else 0.0
        
        obs = np.array([
            self.Vx / 20.0,
            v_err / 5.0,
            np.clip(self.PosY / 0.10, -5.0, 5.0),
            np.clip(self.Vy / 2.0, -5.0, 5.0),
            np.clip(self.phi / 0.10, -5.0, 5.0),
            np.clip(self.gamma / 0.30, -5.0, 5.0),
            np.clip(self.ay_mem / 2.0, -5.0, 5.0),
            np.clip(max_excess / 0.15, 0.0, 5.0),
            in_dwpt,
            self.PosX / 2022.0
        ], dtype=np.float32)
        return obs
        
    def step(self, action):
        """
        action: [a_throttle, a_bias, a_yaw] in [-1, 1]^3
        """
        # Clamp action to [-1, 1]
        a = np.clip(action, -1.0, 1.0)
        alpha_throttle = 1.0 + 0.20 * a[0]    # [0.8, 1.2]
        r_front = 0.50 + 0.15 * a[1]          # [0.35, 0.65]
        delta_Nz_rl = 500.0 * a[2]            # [-500, 500] Nm
        
        r_rear = 1.0 - r_front
        
        step_Phi1 = 0.0
        step_Phi2 = 0.0
        step_Phi3 = 0.0
        step_Phi4 = 0.0
        
        sim = self.sim
        dt = self.dt_micro
        R = self.R
        M, I, lf, lr, d, hg = sim.M, sim.I, sim.lf, sim.lr, sim.d, sim.hg
        l = lf + lr
        Nf_static, Nr_static = sim.Nf, sim.Nr
        
        is_dq = False
        
        for _ in range(self.substeps):
            if self.t >= self.Tsim:
                break
                
            V_ref = sim.get_vref(self.t)
            mu = sim.get_road_friction(self.PosX, self.flag_road)
            F_wind, M_wind = sim.get_wind(self.t, self.flag_wind)
            
            # 1. Velocity Controller
            Delta_V = V_ref - self.Vx
            dv_ff = (V_ref - self.v_ff_state) / sim.tau_ff_v
            self.v_ff_state += dv_ff * dt
            F_ff = (M / sim.tau_ff_v) * (V_ref - self.v_ff_state)
            self.v_integ += Delta_V * dt
            F_fb = sim.Kp_v * Delta_V + sim.Ki_v * self.v_integ
            F_total = np.clip(F_ff + F_fb, -sim.Fmax, sim.Fmax)
            
            # Apply RL throttle modulation
            T_all = F_total * R * alpha_throttle
            
            # --- LATERAL-PRIORITY TRACTION DE-RATING ---
            # Allocates tire friction circle to lateral grip under tracking deviation or slip
            if T_all > 0:
                wheel_speeds = self.w_wheels * R
                current_excess = np.max(wheel_speeds - self.Vx) if self.Vx > 0.1 else 0.0
                slip_metric = current_excess / 0.12
                derate = 1.0 - np.clip(self.lat_derate_slope * (abs(self.PosY) - 0.015) + self.slip_derate_gain * slip_metric * abs(self.PosY), 0.0, 1.0)
                T_all *= max(0.0, derate)
            
            # 2. Dynamic normal load model
            Nfr = Nf_static - (M * self.ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lr / l)) * self.ay_mem
            Nfl = Nf_static - (M * self.ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lr / l)) * self.ay_mem
            Nrr = Nr_static + (M * self.ax_mem * hg) / (2.0 * l) + (M * (hg / d) * (lf / l)) * self.ay_mem
            Nrl = Nr_static + (M * self.ax_mem * hg) / (2.0 * l) - (M * (hg / d) * (lf / l)) * self.ay_mem
            
            # 3. Lateral & Yaw Control (AFS + DYC + RL Yaw Bias + Yaw Damping)
            e_y = 0.0 - self.PosY
            e_phi = 0.0 - self.phi
            
            delta_f_cmd = self.steer_kp_pos * e_y + self.steer_kp_yaw * e_phi
            self.dyc_integ += e_y * dt
            dde = (e_y - self.dyc_deriv_state) / 0.005
            self.dyc_deriv_state += dde * dt
            
            Nz_classical = (self.dyc_kp_pos * e_y + 
                            self.dyc_ki_pos * self.dyc_integ + 
                            self.dyc_kd_pos * dde + 
                            self.dyc_kp_yaw * e_phi)
            
            # High-speed yaw damping: blends in between 10 m/s (36 km/h) and 20 m/s (72 km/h)
            v_blend = np.clip((self.Vx - 10.0) / 10.0, 0.0, 1.0)
            Nz_damp = -self.yaw_damping * v_blend * self.gamma
            
            Nz_cmd = Nz_classical + delta_Nz_rl + Nz_damp
            
            # Smart Directional ESC: clamp only if moment actively destabilizes the vehicle
            if abs(self.gamma) > self.esc_gamma_limit and (Nz_cmd * self.gamma > 0.0):
                scale = max(0.0, 1.0 - (abs(self.gamma) - self.esc_gamma_limit) / 0.10)
                Nz_cmd *= scale
                
            # Slew-rate / low-pass filter to eliminate torque chattering and fishtailing
            if self.tau_dyc > 0:
                self.Nz_filtered += (Nz_cmd - self.Nz_filtered) * (dt / self.tau_dyc)
                Nz_total = self.Nz_filtered
            else:
                Nz_total = Nz_cmd
                
            delta_T = Nz_total * (R / (2.0 * d))
            
            # Clamping DYC delta_T so differential torque does not exceed tire grip on low mu
            if self.dyc_max_dt > 0:
                delta_T = np.clip(delta_T, -self.dyc_max_dt, self.dyc_max_dt)
            
            # 4. Torque distribution with RL front/rear bias
            T_base_f = T_all * r_front * 0.5
            T_base_r = T_all * r_rear * 0.5
            
            T_cmd_raw = np.array([
                T_base_f + delta_T,
                T_base_f - delta_T,
                T_base_r + delta_T,
                T_base_r - delta_T
            ])
            
            # 5. Lateral-Priority Cascade TCS
            lat_priority = np.clip(1.0 - abs(self.PosY) / 0.08, 0.1, 1.0)
            effective_allow_slip = np.maximum(self.slip_speed_min * lat_priority, self.slip_ratio_allow * self.Vx * lat_priority)
            excess_slip = np.maximum(0.0, (self.w_wheels * R - self.Vx) - effective_allow_slip)
            
            T_cmds = np.maximum(0.0, T_cmd_raw - self.K_tcs * excess_slip)
            
            # 6. Loss-Minimization Id* (Morimoto)
            we_f = self.we_f_const * max(0.0, self.Vx)
            we_f_sq = we_f * we_f
            id_f = (self.num_f_k * we_f_sq) / (self.den_f_a + self.den_f_b * we_f_sq)
            
            we_r = self.we_r_const * max(0.0, self.Vx)
            we_r_sq = we_r * we_r
            id_r = (self.num_r_k * we_r_sq) / (self.den_r_a + self.den_r_b * we_r_sq)
            
            # 7. Actuators
            delta_f_cmd = np.clip(delta_f_cmd, -sim.EPS_Del_max, sim.EPS_Del_max)
            self.delta_f_actual += (delta_f_cmd - self.delta_f_actual) * (dt / sim.EPS_tau)
            T_limits = np.array([sim.Tmax_f, sim.Tmax_f, sim.Tmax_r, sim.Tmax_r])
            T_cmds = np.clip(T_cmds, -T_limits, T_limits)
            self.T_actual += (T_cmds - self.T_actual) * (dt / sim.Motor_tau)
            
            # 8. Tire velocities & Pacejka
            V_mag = np.sqrt(self.Vx**2 + self.Vy**2)
            beta = np.arctan2(self.Vy, self.Vx) if self.Vx > 0.05 else 0.0
            gamma_val = self.gamma
            
            V_fl = np.sqrt((V_mag*np.sin(beta) + gamma_val*lf)**2 + (V_mag*np.cos(beta) - gamma_val*d/2.0)**2)
            V_fr = np.sqrt((V_mag*np.sin(beta) + gamma_val*lf)**2 + (V_mag*np.cos(beta) + gamma_val*d/2.0)**2)
            V_rl = np.sqrt((V_mag*np.sin(beta) - gamma_val*lr)**2 + (V_mag*np.cos(beta) - gamma_val*d/2.0)**2)
            V_rr = np.sqrt((V_mag*np.sin(beta) - gamma_val*lr)**2 + (V_mag*np.cos(beta) + gamma_val*d/2.0)**2)
            
            alpha_fl = np.arctan2((V_mag*np.sin(beta) + gamma_val*lf), (V_mag*np.cos(beta) - gamma_val*d/2.0)) - self.delta_f_actual
            alpha_fr = np.arctan2((V_mag*np.sin(beta) + gamma_val*lf), (V_mag*np.cos(beta) + gamma_val*d/2.0)) - self.delta_f_actual
            alpha_rl = np.arctan2((V_mag*np.sin(beta) - gamma_val*lr), (V_mag*np.cos(beta) - gamma_val*d/2.0))
            alpha_rr = np.arctan2((V_mag*np.sin(beta) - gamma_val*lr), (V_mag*np.cos(beta) + gamma_val*d/2.0))
            
            Fx_fr, Fy_fr, lam_fr = sim.pacejka(V_fr, self.w_wheels[0] * R, alpha_fr, mu, Nfr)
            Fx_fl, Fy_fl, lam_fl = sim.pacejka(V_fl, self.w_wheels[1] * R, alpha_fl, mu, Nfl)
            Fx_rr, Fy_rr, lam_rr = sim.pacejka(V_rr, self.w_wheels[2] * R, alpha_rr, mu, Nrr)
            Fx_rl, Fy_rl, lam_rl = sim.pacejka(V_rl, self.w_wheels[3] * R, alpha_rl, mu, Nrl)
            
            self.w_wheels[0] += ((self.T_actual[0] - R * Fx_fr) / sim.Jw_f) * dt
            self.w_wheels[1] += ((self.T_actual[1] - R * Fx_fl) / sim.Jw_f) * dt
            self.w_wheels[2] += ((self.T_actual[2] - R * Fx_rr) / sim.Jw_r) * dt
            self.w_wheels[3] += ((self.T_actual[3] - R * Fx_rl) / sim.Jw_r) * dt
            
            # 9. Vehicle Body Dynamics
            c_del, s_del = np.cos(self.delta_f_actual), np.sin(self.delta_f_actual)
            F_aero = 0.5 * sim.rho * sim.Cd * sim.Afpa * (self.Vx**2)
            F_roll = M * sim.g * sim.mu0
            Fx_tot = (Fx_fl + Fx_fr) * c_del - (Fy_fl + Fy_fr) * s_del + Fx_rl + Fx_rr - F_aero - F_roll
            Fy_tot = (Fx_fl + Fx_fr) * s_del + (Fy_fl + Fy_fr) * c_del + Fy_rl + Fy_rr + F_wind
            
            N_front = lf * ((Fx_fl + Fx_fr) * s_del + (Fy_fl + Fy_fr) * c_del)
            N_rear = -lr * (Fy_rl + Fy_rr)
            N_diff = (d / 2.0) * ((Fx_fr - Fx_fl) * c_del + (Fy_fl - Fy_fr) * s_del + Fx_rr - Fx_rl)
            Nz_tot = N_front + N_rear + N_diff + M_wind
            
            ax = Fx_tot / M
            ay = Fy_tot / M
            self.ax_mem, self.ay_mem = ax, ay
            
            dVx = ax + self.Vy * gamma_val
            dVy = ay - self.Vx * gamma_val
            dgamma = Nz_tot / I
            self.Vx += dVx * dt
            self.Vy += dVy * dt
            self.gamma += dgamma * dt
            
            dPosX = self.Vx * np.cos(self.phi) - self.Vy * np.sin(self.phi)
            dPosY = self.Vx * np.sin(self.phi) + self.Vy * np.cos(self.phi)
            self.PosX += dPosX * dt
            self.PosY += dPosY * dt
            self.phi += gamma_val * dt
            
            # Disqualification
            if abs(self.PosY) > 5.0:
                is_dq = True
                break
                
            # Scoring components
            Pin_fr, _ = sim.calc_motor_power(self.T_actual[0], self.w_wheels[0], True, id_f)
            Pin_fl, _ = sim.calc_motor_power(self.T_actual[1], self.w_wheels[1], True, id_f)
            Pin_rr, _ = sim.calc_motor_power(self.T_actual[2], self.w_wheels[2], False, id_r)
            Pin_rl, _ = sim.calc_motor_power(self.T_actual[3], self.w_wheels[3], False, id_r)
            P_all = Pin_fr + Pin_fl + Pin_rr + Pin_rl
            eta_dwpt = sim.get_dwpt_eta(self.PosX, self.PosY) / 100.0
            
            step_Phi1 += P_all * dt / 3600.0
            step_Phi2 += (lam_fl**2 + lam_fr**2 + lam_rl**2 + lam_rr**2) * dt
            step_Phi3 += (1.0 - eta_dwpt)**2 * dt
            step_Phi4 += ay**2 * dt
            
            self.t += dt
            
        self.step_count += 1
        
        self.cum_Phi1 += step_Phi1
        self.cum_Phi2 += step_Phi2
        self.cum_Phi3 += step_Phi3
        self.cum_Phi4 += step_Phi4
        
        # Immediate step penalty according to official scoring formula:
        # Phi_step = 1*Phi1 + 60*Phi2 + 3*Phi3 + 10*Phi4
        cost_step = (1.0 * step_Phi1 + 
                     60.0 * step_Phi2 + 
                     3.0 * step_Phi3 + 
                     10.0 * step_Phi4)
                     
        # Reward is negative cost (scaled for RL numerical stability)
        reward = -cost_step * 2.0
        
        done = (self.t >= self.Tsim) or is_dq
        if is_dq:
            reward -= 1000.0  # huge penalty for DQ
            
        info = {
            't': self.t,
            'PosX': self.PosX,
            'PosY': self.PosY,
            'Vx': self.Vx,
            'is_dq': is_dq,
            'cum_Phi1': self.cum_Phi1,
            'cum_Phi2': self.cum_Phi2,
            'cum_Phi3': self.cum_Phi3,
            'cum_Phi4': self.cum_Phi4,
            'total_Phi': (1.0*self.cum_Phi1 + 60.0*self.cum_Phi2 + 
                          3.0*self.cum_Phi3 + 10.0*self.cum_Phi4)
        }
        
        return self._get_obs(), reward, done, info
