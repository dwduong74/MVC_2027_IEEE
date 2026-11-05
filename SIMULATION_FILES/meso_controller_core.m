function [delta_f_cmd, T_FR_cmd, Id_FR_cmd, T_FL_cmd, Id_FL_cmd, T_RR_cmd, Id_RR_cmd, T_RL_cmd, Id_RL_cmd] = meso_controller_core(Tall, Sensor)
% MESO_CONTROLLER_CORE (Phase 2 Advanced Optimization)
% IEEE-VTS Motor Vehicle Challenge 2027 (MVC2027)
% Multi-Objective Energy Management & Motion Control Strategy
%
% Integrates:
% 1. Bidirectional Cascade Traction Control (TCS) & Anti-Lock Braking (ABS) (Ueno et al., 2025)
% 2. Full-Cycle Regenerative Braking with 70.5/29.5 Front/Rear Loss-Minimization (LMC)
% 3. Smooth DWPT Centering Barrier with Band-Limited DYC (Morimoto, Fujimoto et al.)
% 4. Adaptive Torque-Dependent d-Axis Current Field Weakening (Morimoto et al., 1994)
%
% Inputs:
%   Tall   : Total required driving/braking torque command [Nm] (from velocity controller)
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
persistent CtlParams last_int_ey dyc_deriv_state

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
        CtlParams.TCS.K_tcs = 800.0;
        CtlParams.TCS.slip_ratio_allow = 0.07;
    end
    try
        CtlParams.ESC = evalin('base', 'ESC');
    catch
        CtlParams.ESC.gamma_limit = 0.25;
    end

    last_int_ey = 0;
    dyc_deriv_state = 0;
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

%% 2. Advanced Multi-Objective MESO Strategy (control_strategy == 2)
% Error states (reference Y = 0, Yaw = 0)
ey = 0 - PosY;
eyaw = 0 - Yaw;

% Simulation time step
Ts = 0.0002; 

% Numerical integration of lateral error with anti-windup
last_int_ey = last_int_ey + ey * Ts;
if abs(last_int_ey) > 0.05
    last_int_ey = sign(last_int_ey) * 0.05;
end

% Exact band-limited derivative: TransferFcn [1 0] / [DYC.Taud_pos 1]
tau_d = CtlParams.DYC.Taud_pos; % 0.005 s
dde = (ey - dyc_deriv_state) / tau_d;
dyc_deriv_state = dyc_deriv_state + dde * Ts;

% --- A. Active Front Steering (Smooth, Proven PD Law) ---
delta_f_cmd = CtlParams.Steer.Kp_pos * ey + CtlParams.Steer.Kp_yaw * eyaw;
% Physical limit clamp (Del_max = pi/60 approx 0.0524 rad = 3 deg)
if abs(delta_f_cmd) > 0.0524
    delta_f_cmd = sign(delta_f_cmd) * 0.0524;
end

% --- B. Direct Yaw Moment Control (DYC) with DWPT Precision Centering ---
% Lateral PID + Heading P control (Exact match to official structure)
Ki_dyc = 35.0;
if isfield(CtlParams.DYC, 'Ki_pos')
    Ki_dyc = CtlParams.DYC.Ki_pos;
end
Nz_pos = CtlParams.DYC.Kp_pos * ey + Ki_dyc * last_int_ey + CtlParams.DYC.Kd_pos * dde;
Nz_yaw = CtlParams.DYC.Kp_yaw * eyaw;

% Smooth Progressive DWPT Barrier: keeps |PosY| <= 0.050m (safely within 0.070m cliff)
if abs(ey) > 0.035
    excess_y = abs(ey) - 0.035;
    Nz_barrier = sign(ey) * 8000.0 * excess_y;
else
    Nz_barrier = 0.0;
end

Nz = Nz_pos + Nz_yaw + Nz_barrier;

% Directional ESC Barrier: limit extreme yaw rate to prevent spinout
if abs(gamma) > CtlParams.ESC.gamma_limit && (Nz * gamma > 0)
    Nz = Nz * max(0, 1.0 - (abs(gamma) - CtlParams.ESC.gamma_limit) / 0.10);
end

% Compute differential torque between right and left wheels
% Differential moment: Nz = (d/R) * ( (T_FR + T_RR) - (T_FL + T_RL) )
R = CtlParams.Wheel.R;
d = CtlParams.Vehicle.d;
delta_T = Nz * (R / (2.0 * d));

% Clamp differential torque per side to prevent wheel overload
delta_T_max = 250;
if abs(delta_T) > delta_T_max
    delta_T = sign(delta_T) * delta_T_max;
end

% --- C. Loss-Minimizing Asymmetric Front/Rear Distribution (LMC) ---
% Front motors: Ra_f = 0.086 Ohm, Psi_f = 0.180 Wb (2.39x more copper-efficient)
% Rear motors:  Ra_r = 0.143 Ohm, Psi_r = 0.125 Wb
% Optimal copper loss ratio: k_front = 0.705, k_rear = 0.295
k_f = 0.705;
k_r = 0.295;

T_FR_base = (k_f / 2.0) * Tall + 0.5 * delta_T;
T_FL_base = (k_f / 2.0) * Tall - 0.5 * delta_T;
T_RR_base = (k_r / 2.0) * Tall + 0.5 * delta_T;
T_RL_base = (k_r / 2.0) * Tall - 0.5 * delta_T;

% --- D. Bidirectional Traction Control (TCS) & Anti-Lock Braking (ABS) ---
% Ref: Ueno, Nguyen et al., IEEE/ASME T-Mech 2025
slip_allow_accel = 0.07; % 7% allowable acceleration slip
slip_allow_brake = 0.06; % 6% allowable braking slip
K_tcs = 800.0;           % Proportional clamping gain [Nm/(m/s)]

% Allowable wheel linear velocity
V_base = max(abs(Vx), 0.5);
delta_V_allow_accel = max(0.12, slip_allow_accel * V_base);
delta_V_allow_brake = max(0.12, slip_allow_brake * V_base);

% Individual wheel linear speeds
Vw_FR = abs(w_FR) * R;
Vw_FL = abs(w_FL) * R;
Vw_RR = abs(w_RR) * R;
Vw_RL = abs(w_RL) * R;

if Tall >= 0
    % --- 1. ACCELERATION / TRACTION MODE (Tall >= 0) ---
    % Excess spin velocities (wheel spinning faster than vehicle)
    excess_FR = max(0.0, (Vw_FR - Vx) - delta_V_allow_accel);
    excess_FL = max(0.0, (Vw_FL - Vx) - delta_V_allow_accel);
    excess_RR = max(0.0, (Vw_RR - Vx) - delta_V_allow_accel);
    excess_RL = max(0.0, (Vw_RL - Vx) - delta_V_allow_accel);

    % Subtractive TCS torque reduction
    T_FR = max(0.0, T_FR_base - K_tcs * excess_FR);
    T_FL = max(0.0, T_FL_base - K_tcs * excess_FL);
    T_RR = max(0.0, T_RR_base - K_tcs * excess_RR);
    T_RL = max(0.0, T_RL_base - K_tcs * excess_RL);
else
    % --- 2. DECELERATION / REGENERATIVE BRAKING MODE (Tall < 0) ---
    % Recover kinetic energy: T_base is negative.
    % Excess lockup velocities (wheel spinning slower than vehicle by > 6%)
    excess_FR_lock = max(0.0, (Vx - Vw_FR) - delta_V_allow_brake);
    excess_FL_lock = max(0.0, (Vx - Vw_FL) - delta_V_allow_brake);
    excess_RR_lock = max(0.0, (Vx - Vw_RR) - delta_V_allow_brake);
    excess_RL_lock = max(0.0, (Vx - Vw_RL) - delta_V_allow_brake);

    % Anti-lock relief: if lockup occurs, add positive torque (reduce braking magnitude)
    T_FR = min(0.0, T_FR_base + K_tcs * excess_FR_lock);
    T_FL = min(0.0, T_FL_base + K_tcs * excess_FL_lock);
    T_RR = min(0.0, T_RR_base + K_tcs * excess_RR_lock);
    T_RL = min(0.0, T_RL_base + K_tcs * excess_RL_lock);

    % Low-speed anti-reversal cutoff (smooth ramp down when stopping)
    if abs(Vx) < 0.3
        ramp = max(0.0, abs(Vx) / 0.3);
        T_FR = T_FR * ramp;
        T_FL = T_FL * ramp;
        T_RR = T_RR * ramp;
        T_RL = T_RL * ramp;
    end
end

% Motor physical saturation limits
T_FR_cmd = min(max(T_FR, -CtlParams.Motor.Tmax_f), CtlParams.Motor.Tmax_f);
T_FL_cmd = min(max(T_FL, -CtlParams.Motor.Tmax_f), CtlParams.Motor.Tmax_f);
T_RR_cmd = min(max(T_RR, -CtlParams.Motor.Tmax_r), CtlParams.Motor.Tmax_r);
T_RL_cmd = min(max(T_RL, -CtlParams.Motor.Tmax_r), CtlParams.Motor.Tmax_r);

% --- E. Adaptive Field-Weakening d-Axis Currents ---
% S. Morimoto et al., IEEE T-IE, 1994
Rce_f = 300.13; Ra_f = 0.086; Ld_f = 0.00063; Psi_f = 0.180; pn_f = 10;
Rce_r = 300.15; Ra_r = 0.143; Ld_r = 0.00130; Psi_r = 0.125; pn_r = 12;

we_f = pn_f * max(0.0, Vx / R);
we_r = pn_r * max(0.0, Vx / R);

num_f = -(Ra_f + Rce_f) * Ld_f * Psi_f * (we_f^2);
den_f = Ra_f * (Rce_f^2) + (Ra_f + Rce_f) * ((Ld_f * we_f)^2);
if den_f > 0
    id_f = num_f / den_f;
else
    id_f = 0.0;
end

num_r = -(Ra_r + Rce_r) * Ld_r * Psi_r * (we_r^2);
den_r = Ra_r * (Rce_r^2) + (Ra_r + Rce_r) * ((Ld_r * we_r)^2);
if den_r > 0
    id_r = num_r / den_r;
else
    id_r = 0.0;
end

% Clamp Id commands to physical operating region
Id_FR_cmd = max(min(id_f, 0.0), -60.0);
Id_FL_cmd = max(min(id_f, 0.0), -60.0);
Id_RR_cmd = max(min(id_r, 0.0), -60.0);
Id_RL_cmd = max(min(id_r, 0.0), -60.0);

end
