"""
Pseudo-Secret Driving Cycle Generator for MVC 2027
====================================================
Tạo các kịch bản giả bí mật để kiểm tra tính tổng quát của giải pháp.

Domain Randomization Parameters:
  - mu_low: hệ số ma sát thấp [0.15, 0.30]
  - mu_zone: vị trí và chiều dài vùng trơn trượt
  - wind: thời điểm, cường độ, hướng gió ngang
  - dwpt_shift: dịch chuyển vùng DWPT
  - tsim: thời gian mô phỏng khác nhau

Each scenario overrides the KanonSimulator defaults.
"""

import numpy as np
import sys
import os

# Add parent TRAINING dir to path
sys.path.insert(0, os.path.dirname(__file__))
from env_kanon_vehicle import KanonSimulator
from improved_classical_control import ImprovedController, run_full_simulation


class PseudoSecretCycle:
    """
    A driving cycle with randomized parameters.
    Can override any aspect of the standard 4 conditions.
    """
    
    def __init__(self, seed=None, **kwargs):
        rng = np.random.default_rng(seed)
        
        # ---- Road Friction Parameters ----
        self.flag_road = kwargs.get('flag_road', int(rng.choice([0, 1])))
        self.mu_low    = kwargs.get('mu_low', float(rng.uniform(0.15, 0.30)))
        self.mu_high   = kwargs.get('mu_high', float(rng.uniform(0.75, 0.95)))
        self.mu_mid    = kwargs.get('mu_mid',  float(rng.uniform(0.45, 0.70)))
        
        # Low-mu zone position (can be anywhere from 5m to 100m)
        L_in_base = kwargs.get('L_in', float(rng.uniform(5.0, 80.0)))
        L_len = kwargs.get('L_len', float(rng.uniform(20.0, 60.0)))
        self.L_in  = L_in_base
        self.L_out = L_in_base + L_len
        
        # Second mu-change zone (can be near end of track)
        M_in_base = kwargs.get('M_in', float(rng.uniform(1500.0, 1950.0)))
        M_len = kwargs.get('M_len', float(rng.uniform(20.0, 50.0)))
        self.M_in  = M_in_base
        self.M_out = M_in_base + M_len
        
        # ---- Wind Parameters ----
        self.flag_wind     = kwargs.get('flag_wind', int(rng.choice([0, 1])))
        self.wind_start    = kwargs.get('wind_start', float(rng.uniform(20.0, 80.0)))
        self.wind_duration = kwargs.get('wind_duration', float(rng.uniform(5.0, 20.0)))
        self.wind_stop     = self.wind_start + self.wind_duration
        self.wind_force    = kwargs.get('wind_force', float(rng.uniform(800.0, 2500.0)))
        self.wind_moment   = kwargs.get('wind_moment', self.wind_force * 0.1)
        self.wind_dir      = kwargs.get('wind_dir', rng.choice([-1, 1]))  # +1 or -1 (dir)
        
        # ---- DWPT Zone Parameters ----
        # Standard: X=500 to X=1500, coil pitch 2m
        # Secret: might start earlier or end later
        self.dwpt_x_start = kwargs.get('dwpt_x_start', float(rng.uniform(300.0, 600.0)))
        self.dwpt_x_end   = kwargs.get('dwpt_x_end',   float(rng.uniform(1400.0, 1700.0)))
        
        # ---- Velocity Profile Parameters ----
        # Allow slightly different velocity profile
        self.v_max_1  = kwargs.get('v_max_1', float(rng.uniform(12.0, 18.0)))
        self.v_max_2  = kwargs.get('v_max_2', float(rng.uniform(3.0, 7.0)))
        
        # ---- Simulation Time ----
        self.Tsim = kwargs.get('Tsim', float(rng.uniform(100.0, 140.0)))
    
    def describe(self, indent="  "):
        lines = [
            f"{indent}Road: flag={self.flag_road}",
            f"{indent}  mu_low={self.mu_low:.3f} at X=[{self.L_in:.1f}, {self.L_out:.1f}]m",
            f"{indent}  mu_high={self.mu_high:.3f}, mu_mid={self.mu_mid:.3f}",
            f"{indent}  mu_mid_zone X=[{self.M_in:.1f}, {self.M_out:.1f}]m",
            f"{indent}Wind: flag={self.flag_wind}",
            f"{indent}  t=[{self.wind_start:.1f}, {self.wind_stop:.1f}]s",
            f"{indent}  F={self.wind_force:.0f}N, dir={'+' if self.wind_dir>0 else '-'}",
            f"{indent}DWPT: X=[{self.dwpt_x_start:.0f}, {self.dwpt_x_end:.0f}]m",
            f"{indent}V_max: {self.v_max_1:.1f}+{self.v_max_2:.1f} m/s, Tsim={self.Tsim:.0f}s",
        ]
        return "\n".join(lines)


def run_simulation_with_cycle(cycle: PseudoSecretCycle, 
                               controller_params=None,
                               verbose=False):
    """
    Run simulation with a custom PseudoSecretCycle.
    
    This overrides the KanonSimulator's internal road/wind/DWPT getters
    to use the cycle's randomized parameters.
    """
    sim = KanonSimulator()
    ctrl = ImprovedController(controller_params)
    ctrl.reset()
    
    # Override simulation time
    Tsim = cycle.Tsim
    Ts = sim.Ts
    R = sim.R
    
    # ----- Monkey-patch sim methods -----
    def custom_road_friction(Pos_X):
        if not cycle.flag_road:
            return cycle.mu_high
        if cycle.L_in <= Pos_X <= cycle.L_out:
            return cycle.mu_low
        elif cycle.M_in <= Pos_X <= cycle.M_out:
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
    
    def custom_dwpt_eta(Pos_X, Pos_Y):
        # Only active in custom DWPT zone
        if Pos_X < cycle.dwpt_x_start or Pos_X > cycle.dwpt_x_end:
            return 0.0
        return sim.get_dwpt_eta(Pos_X, Pos_Y)
    
    # State variables (identical to run_full_simulation)
    Vx = 0.0
    Vy = 0.0
    YawRate = 0.0
    Yaw = 0.0
    Pos_X = 0.0
    Pos_Y = 0.0
    beta = 0.0
    T_act = [0.0, 0.0, 0.0, 0.0]
    w_wheels = [0.0, 0.0, 0.0, 0.0]
    delta_f_act = 0.0
    
    Phi1_acc = 0.0
    Phi2_acc = 0.0
    Phi3_acc = 0.0
    Phi4_acc = 0.0
    
    v_int_err = 0.0
    n_steps = int(Tsim / Ts)
    
    class VehicleRef:
        pass
    veh = VehicleRef()
    veh.d = sim.d; veh.lf = sim.lf; veh.lr = sim.lr
    veh.l = sim.l; veh.R = R
    
    for step in range(n_steps):
        t = step * Ts
        
        Vref = custom_vref(t)
        v_err = Vref - Vx
        v_int_err += v_err * Ts
        a_ref = (custom_vref(t) - custom_vref(t - sim.tau_ff_v)) / sim.tau_ff_v if t > 0 else 0.0
        T_all_cmd = np.clip(sim.Kp_v * v_err + sim.Ki_v * v_int_err + sim.M * a_ref * R,
                             0.0, sim.Fmax * R)
        
        delta_cmd = ctrl.compute_steering(Pos_Y, Yaw)
        delta_cmd = np.clip(delta_cmd, -sim.EPS_Del_max, sim.EPS_Del_max)
        Nz = ctrl.compute_dyc(Pos_Y, Yaw, Ts)
        
        T_FR_raw, T_FL_raw, T_RR_raw, T_RL_raw = ctrl.distribute_torques(T_all_cmd, Nz, veh)
        T_FR_cmd, T_FL_cmd, T_RR_cmd, T_RL_cmd = ctrl.apply_tcs(
            T_FR_raw, T_FL_raw, T_RR_raw, T_RL_raw,
            w_wheels[0], w_wheels[1], w_wheels[2], w_wheels[3], Vx, R)
        T_FR_cmd = np.clip(T_FR_cmd, 0.0, sim.Tmax_f)
        T_FL_cmd = np.clip(T_FL_cmd, 0.0, sim.Tmax_f)
        T_RR_cmd = np.clip(T_RR_cmd, 0.0, sim.Tmax_r)
        T_RL_cmd = np.clip(T_RL_cmd, 0.0, sim.Tmax_r)
        
        alpha_m = sim.Motor_tau / (sim.Motor_tau + Ts)
        for i, cmd in enumerate([T_FR_cmd, T_FL_cmd, T_RR_cmd, T_RL_cmd]):
            T_act[i] = alpha_m * T_act[i] + (1 - alpha_m) * cmd
        
        alpha_s = sim.EPS_tau / (sim.EPS_tau + Ts)
        delta_f_act = alpha_s * delta_f_act + (1 - alpha_s) * delta_cmd
        
        mu = custom_road_friction(Pos_X)
        F_wind, M_wind = custom_wind(t)
        
        Fz_list = [sim.Nf, sim.Nf, sim.Nr, sim.Nr]
        Jw_list = [sim.Jw_f, sim.Jw_f, sim.Jw_r, sim.Jw_r]
        slip_angles = [delta_f_act - beta, delta_f_act - beta, -beta, -beta]
        
        lambdas = [0.0]*4
        Fx_w = [0.0]*4
        for i in range(4):
            Fxi, Fyi, lam = sim.pacejka(Vx, w_wheels[i]*R, slip_angles[i], mu, Fz_list[i])
            lambdas[i] = lam
            Fx_w[i] = Fxi
            dw = (T_act[i] - R * Fxi) / Jw_list[i]
            w_wheels[i] = max(0.0, w_wheels[i] + dw * Ts)
        
        F_net = sum(Fx_w) - sim.mu0 * sim.M * sim.g - 0.5*sim.rho*sim.Cd*sim.Afpa*Vx**2
        ax = F_net / sim.M
        Vx_new = max(0.0, Vx + ax * Ts)
        
        Cf, Cr = 50000.0, 55000.0
        alpha_f_tire = (delta_f_act - (Vy + sim.lf*YawRate)/max(0.1,Vx)) if Vx>0.5 else 0.0
        alpha_r_tire = (-(Vy - sim.lr*YawRate)/max(0.1,Vx)) if Vx>0.5 else 0.0
        Fyf = Cf * alpha_f_tire
        Fyr = Cr * alpha_r_tire
        ay = (Fyf + Fyr + F_wind) / sim.M
        Vy_new = Vy + (ay - Vx*YawRate) * Ts
        
        Nz_tires = sim.lf*Fyf - sim.lr*Fyr + M_wind
        Nz_dyc = ((T_act[0]+T_act[2]) - (T_act[1]+T_act[3])) * sim.d / (2.0*R)
        YawRate_new = YawRate + (Nz_tires + Nz_dyc)/sim.I * Ts
        
        Pos_X += (Vx*np.cos(Yaw) - Vy*np.sin(Yaw)) * Ts
        Pos_Y += (Vx*np.sin(Yaw) + Vy*np.cos(Yaw)) * Ts
        Yaw += YawRate * Ts
        beta = np.arctan2(Vy, max(0.1, Vx))
        Vx = Vx_new; Vy = Vy_new; YawRate = YawRate_new
        
        # Id* and power
        Id_FR, Id_FL, Id_RR, Id_RL = ctrl.compute_id_stars(
            T_act[0], T_act[1], T_act[2], T_act[3],
            w_wheels[0], w_wheels[1], w_wheels[2], w_wheels[3], sim)
        
        eta_dwpt = custom_dwpt_eta(Pos_X, Pos_Y)
        
        P_in = 0.0
        for T_i, w_i, is_front, Id_i in [
            (T_act[0],w_wheels[0],True, Id_FR),
            (T_act[1],w_wheels[1],True, Id_FL),
            (T_act[2],w_wheels[2],False,Id_RR),
            (T_act[3],w_wheels[3],False,Id_RL)]:
            Pi, _ = sim.calc_motor_power(T_i, w_i, is_front, Id_i)
            P_in += max(0.0, Pi)
        
        Phi1_acc += P_in * Ts
        Phi2_acc += sum(l**2 for l in lambdas) * Ts
        Phi3_acc += (1.0 - eta_dwpt/100.0)**2 * Ts
        Phi4_acc += ay**2 * Ts
        
        if abs(Pos_Y) > 5.0:
            Phi2_acc += 1e6
            if verbose:
                print(f"  [DQ] |Pos_Y|={abs(Pos_Y):.2f}m at t={t:.2f}s!")
            break
    
    sigma = [1, 60, 3, 10]
    Phi1 = Phi1_acc / 3600.0
    Phi = sigma[0]*Phi1 + sigma[1]*Phi2_acc + sigma[2]*Phi3_acc + sigma[3]*Phi4_acc
    
    return {'Phi': Phi, 'Phi1': Phi1, 'Phi2': Phi2_acc,
            'Phi3': Phi3_acc, 'Phi4': Phi4_acc,
            'Pos_X_final': Pos_X, 'Vx_final': Vx}


def generate_scenario_suite(n_scenarios=20, base_seed=42):
    """
    Generate a suite of diverse pseudo-secret scenarios.
    Includes edge cases that might appear in the actual secret test cycles.
    """
    scenarios = []
    
    # Fixed challenging scenarios
    fixed = [
        # Extreme low-mu very early (worst case)
        {'name': 'SECRET-A: Ultra-low mu early',
         'kwargs': {'flag_road':1, 'mu_low':0.15, 'L_in':5.0, 'L_len':50.0,
                    'flag_wind':1, 'wind_start':50.0, 'wind_force':2000.0}},
        # Wind BEFORE DWPT zone
        {'name': 'SECRET-B: Wind before DWPT',
         'kwargs': {'flag_road':1, 'mu_low':0.25, 'L_in':15.0, 'L_len':30.0,
                    'flag_wind':1, 'wind_start':25.0, 'wind_duration':15.0, 'wind_force':1800.0,
                    'dwpt_x_start':400.0, 'dwpt_x_end':1600.0}},
        # DWPT zone VERY early  
        {'name': 'SECRET-C: DWPT starts very early',
         'kwargs': {'flag_road':1, 'mu_low':0.20, 'L_in':10.0, 'L_len':40.0,
                    'flag_wind':1, 'wind_start':60.0, 'wind_force':1600.0,
                    'dwpt_x_start':200.0, 'dwpt_x_end':1200.0}},
        # Double low-mu zones
        {'name': 'SECRET-D: Double mu-drop zones',
         'kwargs': {'flag_road':1, 'mu_low':0.22, 'L_in':10.0, 'L_len':25.0,
                    'M_in':800.0, 'M_in_as_low': True,  # second low-mu
                    'flag_wind':0}},
        # Strong crosswind during braking
        {'name': 'SECRET-E: Strong wind during deceleration',
         'kwargs': {'flag_road':0, 'flag_wind':1,
                    'wind_start':105.0, 'wind_duration':10.0, 'wind_force':2500.0}},
        # Longer simulation
        {'name': 'SECRET-F: Extended 140s simulation',
         'kwargs': {'flag_road':1, 'mu_low':0.25, 'L_in':15.0, 'L_len':30.0,
                    'flag_wind':1, 'wind_start':50.0, 'wind_force':1600.0,
                    'Tsim':140.0, 'dwpt_x_end':1800.0}},
    ]
    
    for f in fixed:
        # Create PseudoSecretCycle from fixed kwargs
        cycle = PseudoSecretCycle(seed=0, **f['kwargs'])
        scenarios.append({'name': f['name'], 'cycle': cycle})
    
    # Random scenarios
    rng = np.random.default_rng(base_seed)
    for i in range(n_scenarios):
        seed = int(rng.integers(1, 10000))
        cycle = PseudoSecretCycle(seed=seed)
        scenarios.append({'name': f'RANDOM-{i+1:02d} (seed={seed})', 'cycle': cycle})
    
    return scenarios


def run_scenario_suite(n_random=10, controller_params=None):
    """Run all scenarios and report statistics"""
    scenarios = generate_scenario_suite(n_scenarios=n_random)
    
    print("=" * 70)
    print("  Pseudo-Secret Cycle Stress Test - MVC 2027")
    print("=" * 70)
    
    results = []
    for s in scenarios:
        cycle = s['cycle']
        result = run_simulation_with_cycle(cycle, controller_params, verbose=False)
        results.append({'name': s['name'], **result})
        
        status = "DQ" if result['Phi2'] > 1e5 else "OK"
        print(f"\n[{status}] {s['name']}")
        print(cycle.describe())
        print(f"  → Phi={result['Phi']:.1f} | Phi1={result['Phi1']:.1f}Wh "
              f"| Phi2(x60)={60*result['Phi2']:.1f} "
              f"| Phi3(x3)={3*result['Phi3']:.1f} "
              f"| Phi4(x10)={10*result['Phi4']:.2f}")
    
    # Statistics
    phis = [r['Phi'] for r in results if r['Phi'] < 1e5]
    if phis:
        print(f"\n{'='*70}")
        print(f"  Statistics over {len(phis)} valid scenarios:")
        print(f"  Mean Phi = {np.mean(phis):.1f}")
        print(f"  Max  Phi = {np.max(phis):.1f}")
        print(f"  Min  Phi = {np.min(phis):.1f}")
        print(f"  Std  Phi = {np.std(phis):.1f}")
        dq_count = sum(1 for r in results if r['Phi'] > 1e5)
        print(f"  DQ count = {dq_count}/{len(results)}")
        print(f"{'='*70}")
    
    return results


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    results = run_scenario_suite(n_random=n)
