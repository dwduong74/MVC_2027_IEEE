import numpy as np
from scipy.interpolate import RegularGridInterpolator

class KanonSimulator:
    def __init__(self):
        # 1. Constants
        self.g = 9.81
        self.eps = 0.05
        
        # 2. Vehicle Body Parameters
        self.M = 880.0
        self.I = 617.0
        self.lf = 0.988
        self.lr = 0.712
        self.l = self.lf + self.lr
        self.d = 1.3
        self.df = self.d
        self.dr = self.d
        self.hg = 0.46
        self.mu0 = 0.0128
        self.rho = 1.205
        self.Cd = 0.48
        self.Afpa = 2.4
        self.Nf = (self.M / 2.0) * self.g * self.lr / self.l
        self.Nr = (self.M / 2.0) * self.g * self.lf / self.l
        
        # 3. Wheel Parameters
        self.R = 0.302
        self.Jw_f = 1.24
        self.Jw_r = 1.26
        
        # 4. Steering Model
        self.EPS_tau = 0.01
        self.EPS_Del_max = (np.pi / 3.0) / 20.0
        
        # 5. Motor Parameters
        self.Tmax_f = 500.0
        self.Tmax_r = 530.0
        self.Ra_f = 0.086
        self.Ra_r = 0.143
        self.Rce_f = 300.13
        self.Rce_r = 300.15
        self.Psi_f = 0.18
        self.Psi_r = 0.125
        self.pn_f = 10
        self.pn_r = 12
        self.Lq_f = 0.00069
        self.Lq_r = 0.00150
        self.Ld_f = 0.00063
        self.Ld_r = 0.00130
        self.Motor_tau = 0.01
        
        # 6. Velocity Controller Parameters
        self.rho_ctl = -0.8
        self.Kp_v = -2.0 * self.rho_ctl * self.M
        self.Ki_v = (self.rho_ctl ** 2) * self.M
        self.tau_ff_v = 0.01
        self.Fmax = 2.0 * (self.Tmax_f + self.Tmax_r) / self.R
        
        # 7. Simulation Settings
        self.Tsim = 120.0
        self.Ts = 0.0002
        
        # 8. DWPT Parameters
        self.WPT_freq = 85000.0
        self.WPT_L1 = 142.326e-6
        self.WPT_L2 = 54.2047e-6
        self.WPT_R1 = 85.02 / 1000.0
        self.WPT_R2 = 39.88 / 1000.0
        self.WPT_P = 10000.0
        self.w_wpt = 2.0 * np.pi * self.WPT_freq
        self.Q1 = self.w_wpt * self.WPT_L1 / self.WPT_R1
        self.Q2 = self.w_wpt * self.WPT_L2 / self.WPT_R2
        
        self.dwpt_x = np.array([-400, -350, -300, -250, -200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300, 350, 400], dtype=float)
        self.dwpt_y = np.array([-150, -100, -50, 0, 50, 100, 150], dtype=float)
        z000 = np.array([18.366, 18.12075, 18.2285, 18.62175, 19.4505, 22.78625, 21.99725, 18.1625, 12.584])
        z050 = np.array([16.16075, 15.495, 15.44675, 15.71575, 18.70275, 19.3575, 18.8615, 15.5565, 10.6605])
        z100 = np.array([9.85675, 9.60925, 10.05325, 10.28375, 11.30575, 13.352, 13.16575, 10.59525, 6.935])
        z150 = np.array([4.81875, 4.1325, 4.1535, 5.24425, 6.22225, 7.28925, 6.64025, 5.4515, 3.44175])
        
        z000f = np.flip(z000)[:-1]
        z050f = np.flip(z050)[:-1]
        z100f = np.flip(z100)[:-1]
        z150f = np.flip(z150)[:-1]
        
        zz = np.array([
            np.concatenate([z150f, z150]),
            np.concatenate([z100f, z100]),
            np.concatenate([z050f, z050]),
            np.concatenate([z000f, z000]),
            np.concatenate([z050f, z050]),
            np.concatenate([z100f, z100]),
            np.concatenate([z150f, z150])
        ])
        self.dwpt_interp = RegularGridInterpolator((self.dwpt_y, self.dwpt_x), zz, bounds_error=False, fill_value=0.0)

    def get_vref(self, t):
        # Ramp 1: slope 2.5, start 0, sat [-15, 15]
        r1 = np.clip(2.5 * max(0.0, t), -15.0, 15.0)
        # Ramp 2: slope 1.0, start 40, sat [-5, 5]
        r2 = np.clip(1.0 * max(0.0, t - 40.0), -5.0, 5.0)
        # Ramp 3: slope 2.5, start 110, sat [-20, 20]
        r3 = np.clip(2.5 * max(0.0, t - 110.0), -20.0, 20.0)
        return r1 + r2 - r3

    def get_road_friction(self, Pos_X, flag_road=1):
        if flag_road == 0:
            return 0.85
        if 12.5 <= Pos_X <= 42.5:
            return 0.25
        elif 1976.99 <= Pos_X <= 2011.03:
            return 0.60
        else:
            return 0.85

    def get_wind(self, t, flag_wind=1):
        if flag_wind == 1 and 45.0 <= t <= 55.0:
            return 1600.0, 160.0
        return 0.0, 0.0

    def pacejka(self, V, Vw, alpha, mu, Fz):
        abs_V = abs(V)
        if abs_V < 0.05:
            if abs(Vw) < 0.05:
                return 0.0, 0.0, 0.0
            else:
                B, C, D, E = 10.0, 1.9, 1.0, 0.97
                tmp1 = B * (1.0 - E) * 1.0 + E * np.arctan(B * 1.0)
                tmp2 = C * np.arctan(tmp1)
                Fx = D * np.sin(tmp2) * mu * Fz
                return Fx, 0.0, 1.0
        else:
            c_a, s_a = np.cos(alpha), np.sin(alpha)
            V_vec = np.array([c_a * V, s_a * V])
            Vw_vec = np.array([Vw, 0.0])
            Vs = Vw_vec - V_vec
            
            norm_V = np.linalg.norm(V_vec)
            norm_Vw = np.linalg.norm(Vw_vec)
            max_norm = max(norm_V, norm_Vw)
            
            lambda_vec = Vs / max_norm if max_norm > 0 else np.zeros(2)
            norm_lambda = min(1.0, np.linalg.norm(lambda_vec))
                
            if abs(Vs[0]) < 1e-6:
                lambda_x = 0.0
                Fx = 0.0
                Fy = 0.0
            else:
                max_abs_vx = max(abs(V_vec[0]), abs(Vw_vec[0]))
                lambda_x = Vs[0] / max_abs_vx if max_abs_vx > 0 else 0.0
                if abs(lambda_x) > 1.0:
                    lambda_x = np.sign(lambda_x) * 1.0
                    
                B, C, D, E = 10.0, 1.9, 1.0, 0.97
                tmp1 = B * (1.0 - E) * norm_lambda + E * np.arctan(B * norm_lambda)
                tmp2 = C * np.arctan(tmp1)
                norm_Vs = np.linalg.norm(Vs)
                F = mu * Fz * D * np.sin(tmp2) * (Vs / norm_Vs) if norm_Vs > 0 else np.zeros(2)
                Fx, Fy = F[0], F[1]
                
            return Fx, Fy, lambda_x

    def get_dwpt_eta(self, Pos_X, Pos_Y):
        u = Pos_X
        if u <= 500.1:
            delta_x = -0.4
        elif u >= 1498.9:
            delta_x = 0.4
        else:
            delta_x = -0.4
            # Optimized lookup for coil
            coil_idx = int(round((u - 500.5) / 2.0))
            if 0 <= coil_idx <= 499:
                pos_coil = 500.5 + 2.0 * coil_idx
                dx = u - pos_coil
                if abs(dx) < 0.4:
                    delta_x = dx
        u1 = delta_x * 1000.0 # mm
        u2 = Pos_Y * 1000.0   # mm
        
        if abs(u1) > 380.0 or abs(u2) > 70.0:
            return 0.0
        
        Lm = 1e-6 * float(self.dwpt_interp((u2, u1)))
        k = Lm / np.sqrt(self.WPT_L1 * self.WPT_L2)
        k2 = k * k
        denom = (1.0 + np.sqrt(1.0 + self.Q1 * self.Q2 * k2)) ** 2
        eta_wpt = (k2 * self.Q1 * self.Q2) / denom if denom > 0 else 0.0
        return eta_wpt * 100.0

    def calc_motor_power(self, T, omega, is_front, Id):
        if is_front:
            Ra = self.Ra_f
            Rce = self.Rce_f
            Psi_a = self.Psi_f
            pn = self.pn_f
            Ld = self.Ld_f
            Lq = self.Lq_f
        else:
            Ra = self.Ra_r
            Rce = self.Rce_r
            Psi_a = self.Psi_r
            pn = self.pn_r
            Ld = self.Ld_r
            Lq = self.Lq_r
            
        Psi_eq = Psi_a + (Ld - Lq) * Id
        Iq = T / (1.5 * pn * Psi_eq) if abs(Psi_eq) > 1e-6 else 0.0
        P_cu = Ra * (Id**2 + Iq**2)
        we = pn * omega
        # Rc calculation
        Rc = Rce
        P_fe = ((we * Psi_eq)**2 + (we * Lq * Iq)**2) / Rc
        P_out = T * omega
        P_in = P_out + P_cu + P_fe
        return P_in, P_out
