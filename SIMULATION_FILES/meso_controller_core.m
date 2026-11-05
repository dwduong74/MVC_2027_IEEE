function [delta_f_cmd, T_FR_cmd, Id_FR_cmd, T_FL_cmd, Id_FL_cmd, T_RR_cmd, Id_RR_cmd, T_RL_cmd, Id_RL_cmd] = meso_controller_core(Tall, Sensor)
% MESO_CONTROLLER_CORE
% IEEE-VTS Motor Vehicle Challenge 2027 (MVC2027)
% Multi-Objective Energy Management & Motion Control Strategy
%
% Integrates:
% 1. Cascade Wheel-Velocity Traction Control System (TCS) (Ueno et al., 2025)
% 2. Optimal 70.5/29.5 Asymmetric Front/Rear Torque Distribution (LMC) (Nguyen et al., 2023)
% 3. Disturbance Observer (DOB) Feedforward Lateral Wind Rejection for DWPT (Koishi et al., 2025)
% 4. Adaptive Torque-Dependent d-Axis Current Field Weakening (Morimoto et al., 1994)
%
% Inputs:
%   Tall   : Total required driving torque command [Nm] (from velocity controller)
%   Sensor : 12-element vector of vehicle measurements:
%            [1] gamma   - Yaw rate [rad/s]
%            [2] ax      - Longitudinal acceleration [m/s^2]
%            [3] ay      - Lateral acceleration [m/s^2]
%            [4] delta_f - Actual front steering angle [rad]
%            [5] w_FR    - Front-Right wheel angular speed [rad/s]
%            [6] w_FL    - Front-Left wheel angular speed [rad/s]
%            [7] w_RR    - Rear-Right wheel angular speed [rad/s]
%            [8] w_RL    - Rear-Left wheel angular speed [rad/s]
%            [9] Vx      - Vehicle longitudinal speed [m/s]
%            [10] beta   - Vehicle side slip angle [rad]
%            [11] PosY   - Lateral position deviation Y [m] (reference = 0)
%            [12] Yaw    - Vehicle yaw / heading angle [rad] (reference = 0)
%
% Outputs:
%   delta_f_cmd : Front steering angle command [rad]
%   T_FR_cmd, T_FL_cmd, T_RR_cmd, T_RL_cmd : Individual wheel motor torques [Nm]
%   Id_FR_cmd, Id_FL_cmd, Id_RR_cmd, Id_RL_cmd : Motor d-axis currents [A]

% Persistent states for fast simulation execution
persistent CtlParams last_int_ey last_ey last_time

if isempty(CtlParams)
    % Cache parameters from workspace on first step
    try
        CtlParams.strategy = evalin('base', 'control_strategy');
    catch
        CtlParams.strategy = 2;
    end
    try
        CtlParams.Vehicle = evalin('base', 'Vehicle');
        CtlParams.Wheel   = evalin('base', 'Wheel');
        CtlParams.Motor   = evalin('base', 'Motor');
        CtlParams.Steer   = evalin('base', 'Steer');
        CtlParams.DYC     = evalin('base', 'DYC');
    catch
        % Safe defaults matching sim_para.m if not yet initialized
        CtlParams.Vehicle.M = 880; CtlParams.Vehicle.d = 1.3;
        CtlParams.Wheel.R = 0.302;
        CtlParams.Motor.Tmax_f = 500; CtlParams.Motor.Tmax_r = 530;
        CtlParams.Steer.Kp_pos = 0.02; CtlParams.Steer.Kp_yaw = 0.2;
        CtlParams.DYC.Kp_pos = 4000; CtlParams.DYC.Ki_pos = 35;
        CtlParams.DYC.Kd_pos = 1000; CtlParams.DYC.Kp_yaw = 5000;
        CtlParams.DYC.Taud_pos = 0.005;
    end
    try
        CtlParams.TCS = evalin('base', 'TCS');
    catch
        CtlParams.TCS.K_tcs = 900.0;
        CtlParams.TCS.slip_ratio_allow = 0.07;
        CtlParams.TCS.lat_derate_slope = 15.0;
    end
    try
        CtlParams.ESC = evalin('base', 'ESC');
    catch
        CtlParams.ESC.gamma_limit = 0.20;
    end

    last_int_ey = 0;
    last_ey = 0;
    last_time = 0;
end

% Extract sensor signals
gamma    = Sensor(1);
ax       = Sensor(2);
ay       = Sensor(3);
delta_f  = Sensor(4);
w_FR     = Sensor(5);
w_FL     = Sensor(6);
w_RR     = Sensor(7);
w_RL     = Sensor(8);
Vx       = Sensor(9);
beta     = Sensor(10);
PosY     = Sensor(11);
Yaw      = Sensor(12);

%% 1. Baseline Strategy (control_strategy == 1)
if CtlParams.strategy == 1
    % Steering angle
    delta_f_cmd = CtlParams.Steer.Kp_pos * (0 - PosY) + CtlParams.Steer.Kp_yaw * (0 - Yaw);
    % Torque distribution (54% Front, 46% Rear, no DYC)
    T_FR_cmd = 0.27 * Tall;
    T_FL_cmd = 0.27 * Tall;
    T_RR_cmd = 0.23 * Tall;
    T_RL_cmd = 0.23 * Tall;
    % D-axis currents = 0
    Id_FR_cmd = 0; Id_FL_cmd = 0; Id_RR_cmd = 0; Id_RL_cmd = 0;
    return;
end

%% 2. Improved Multi-Objective MESO Strategy (control_strategy == 2)
% Error states (reference Y = 0, Yaw = 0)
ey = 0 - PosY;
eyaw = 0 - Yaw;

% Numerical integration of lateral error with anti-windup
Ts = 0.0002; % Simulation time step
last_int_ey = last_int_ey + ey * Ts;
if abs(last_int_ey) > 0.05
    last_int_ey = sign(last_int_ey) * 0.05;
end
dey = (ey - last_ey) / Ts;
last_ey = ey;

% --- A. Active Front Steering with Disturbance Wind Compensation ---
% Baseline PD feedback
delta_f_base = CtlParams.Steer.Kp_pos * ey + CtlParams.Steer.Kp_yaw * eyaw;

% Disturbance Observer: estimate external lateral force
% a_y,nominal = Vx * gamma
a_y_nom = Vx * gamma;
a_y_dist = ay - a_y_nom;
if abs(a_y_dist) > 0.15
    % Wind force estimate: F_wind = M * a_y_dist
    % Cornering stiffness approx: C_f = 25000 N/rad
    delta_f_ff = - (CtlParams.Vehicle.M * a_y_dist) / (2 * 25000);
    % Clamp feedforward to safe range
    if abs(delta_f_ff) > 0.025
        delta_f_ff = sign(delta_f_ff) * 0.025;
    end
else
    delta_f_ff = 0;
end
delta_f_cmd = delta_f_base + delta_f_ff;
% Physical limit clamp (Del_max approx 0.0524 rad = 3 deg)
if abs(delta_f_cmd) > 0.0524
    delta_f_cmd = sign(delta_f_cmd) * 0.0524;
end

% --- B. Direct Yaw Moment Control (DYC) ---
% Lateral PID + Heading P control
Nz_pos = CtlParams.DYC.Kp_pos * ey + CtlParams.DYC.Ki_pos * last_int_ey + CtlParams.DYC.Kd_pos * dey;
Nz_yaw = CtlParams.DYC.Kp_yaw * eyaw;
Nz = Nz_pos + Nz_yaw;

% High-speed yaw damping
Nz = Nz - 2500 * gamma;

% Directional ESC Barrier: limit yaw rate
if abs(gamma) > CtlParams.ESC.gamma_limit && (Nz * gamma > 0)
    Nz = Nz * max(0, 1.0 - (abs(gamma) - CtlParams.ESC.gamma_limit) / 0.10);
end

% Compute differential torque between right and left wheels
R = CtlParams.Wheel.R;
d = CtlParams.Vehicle.d;
delta_T = (R / (2 * d)) * Nz;
% Clamp differential torque per side to prevent wheel overload
delta_T_max = 250;
if abs(delta_T) > delta_T_max
    delta_T = sign(delta_T) * delta_T_max;
end

% --- C. Loss-Minimizing Asymmetric Front/Rear Distribution (LMC) ---
% Front motors: Ra_f = 0.086 Ohm, Psi_f = 0.180 Wb (High efficiency)
% Rear motors:  Ra_r = 0.143 Ohm, Psi_r = 0.125 Wb (Lower efficiency)
% Optimal copper loss ratio: k_front = 0.705, k_rear = 0.295
k_f = 0.705;
k_r = 0.295;

T_FR = (k_f / 2) * Tall + (k_f / 2) * delta_T;
T_FL = (k_f / 2) * Tall - (k_f / 2) * delta_T;
T_RR = (k_r / 2) * Tall + (k_r / 2) * delta_T;
T_RL = (k_r / 2) * Tall - (k_r / 2) * delta_T;

% --- D. Cascade Wheel-Velocity Traction Control System (TCS) ---
% Ref: Ueno, Nguyen et al., IEEE/ASME T-Mech 2025
slip_allow = CtlParams.TCS.slip_ratio_allow; % 0.07 (7%)
K_tcs = CtlParams.TCS.K_tcs;                 % 900 Nm/(m/s)

% Allowable wheel linear velocity
V_base = max(abs(Vx), 0.5);
V_allow = (1 + slip_allow) * V_base;

% Individual wheel anti-slip intervention
Vw_FR = abs(w_FR) * R;
Vw_FL = abs(w_FL) * R;
Vw_RR = abs(w_RR) * R;
Vw_RL = abs(w_RL) * R;

if Vw_FR > V_allow && T_FR > 0
    T_FR = max(0, T_FR - K_tcs * (Vw_FR - V_allow));
end
if Vw_FL > V_allow && T_FL > 0
    T_FL = max(0, T_FL - K_tcs * (Vw_FL - V_allow));
end
if Vw_RR > V_allow && T_RR > 0
    T_RR = max(0, T_RR - K_tcs * (Vw_RR - V_allow));
end
if Vw_RL > V_allow && T_RL > 0
    T_RL = max(0, T_RL - K_tcs * (Vw_RL - V_allow));
end

% Lateral Priority Traction De-rating:
% When vehicle has lateral deviation, reduce driving force to preserve lateral grip
if abs(ey) > 0.03
    lat_derate = max(0.20, 1.0 - CtlParams.TCS.lat_derate_slope * abs(ey));
    T_FR = T_FR * lat_derate;
    T_FL = T_FL * lat_derate;
    T_RR = T_RR * lat_derate;
    T_RL = T_RL * lat_derate;
end

% Motor saturation limits
T_FR_cmd = min(max(T_FR, -CtlParams.Motor.Tmax_f), CtlParams.Motor.Tmax_f);
T_FL_cmd = min(max(T_FL, -CtlParams.Motor.Tmax_f), CtlParams.Motor.Tmax_f);
T_RR_cmd = min(max(T_RR, -CtlParams.Motor.Tmax_r), CtlParams.Motor.Tmax_r);
T_RL_cmd = min(max(T_RL, -CtlParams.Motor.Tmax_r), CtlParams.Motor.Tmax_r);

% --- E. Adaptive Field-Weakening d-Axis Currents ---
% S. Morimoto et al., IEEE T-IE, 1994
% Loss minimization current modulated by vehicle speed and torque demand
Rce_f = 300.13; Ra_f = 0.086; Ld_f = 0.00063; Psi_f = 0.180; pn_f = 10;
Rce_r = 300.15; Ra_r = 0.143; Ld_r = 0.00130; Psi_r = 0.125; pn_r = 12;

% Optimal d-axis current formula from MVC2027 paper
% Id = - (Ra + Rce)*Ld*Psi / [ (Ra+Rce)*Ld^2 + (Rce^2 * Ra / (pn^2 * we^2)) ]
we_f = pn_f * (Vx / R);
we_r = pn_r * (Vx / R);

if we_f > 1.0
    den_f = (Ra_f + Rce_f) * (Ld_f^2) * (pn_f^2) * (we_f^2) + (R*R * Ra_f * Rce_f * Rce_f);
    num_f = -(Ra_f + Rce_f) * Ld_f * Psi_f * (pn_f^2) * (we_f^2) * (R*R);
    Id_f_opt = num_f / max(den_f, 1e-6);
else
    Id_f_opt = 0;
end

if we_r > 1.0
    den_r = (Ra_r + Rce_r) * (Ld_r^2) * (pn_r^2) * (we_r^2) + (R*R * Ra_r * Rce_r * Rce_r);
    num_r = -(Ra_r + Rce_r) * Ld_r * Psi_r * (pn_r^2) * (we_r^2) * (R*R);
    Id_r_opt = num_r / max(den_r, 1e-6);
else
    Id_r_opt = 0;
end

% Clamp Id commands
Id_FR_cmd = max(min(Id_f_opt, 0), -60);
Id_FL_cmd = max(min(Id_f_opt, 0), -60);
Id_RR_cmd = max(min(Id_r_opt, 0), -60);
Id_RL_cmd = max(min(Id_r_opt, 0), -60);

end
