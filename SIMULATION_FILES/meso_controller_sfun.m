function meso_controller_sfun(block)
% MESO_CONTROLLER_SFUN
% Level-2 MATLAB S-Function interface for the Multi-Objective MESO Controller
% Compatible with MATLAB R2022b+, MATLAB Online, and all Simulink installations.

setup(block);

% =========================================================================
% S-Function Setup
% =========================================================================
function setup(block)

  % Register number of input and output ports
  block.NumInputPorts  = 2; % Port 1: Tall (1), Port 2: Sensor (12)
  block.NumOutputPorts = 9; % 9 control commands

  % Configure Input Port 1: Tall (Scalar total driving command)
  block.InputPort(1).Dimensions        = 1;
  block.InputPort(1).DatatypeID        = 0; % double
  block.InputPort(1).Complexity        = 'Real';
  block.InputPort(1).DirectFeedthrough = true;

  % Configure Input Port 2: Sensor (12-element measurement vector)
  block.InputPort(2).Dimensions        = 12;
  block.InputPort(2).DatatypeID        = 0; % double
  block.InputPort(2).Complexity        = 'Real';
  block.InputPort(2).DirectFeedthrough = true;

  % Configure 9 Output Ports (All scalars, double, real)
  % Out 1: delta_f_cmd
  % Out 2: T_FR_cmd, Out 3: Id_FR_cmd
  % Out 4: T_FL_cmd, Out 5: Id_FL_cmd
  % Out 6: T_RR_cmd, Out 7: Id_RR_cmd
  % Out 8: T_RL_cmd, Out 9: Id_RL_cmd
  for i = 1:9
      block.OutputPort(i).Dimensions   = 1;
      block.OutputPort(i).DatatypeID   = 0;
      block.OutputPort(i).Complexity   = 'Real';
  end

  % Continuous sample time / inherited from solver
  block.SampleTimes = [0 0];

  % Register block methods
  block.SimStateCompliance = 'DefaultSimState';
  block.RegBlockMethod('Outputs',   @Outputs);
  block.RegBlockMethod('Terminate', @Terminate);

% =========================================================================
% Compute Outputs
% =========================================================================
function Outputs(block)

  Tall   = block.InputPort(1).Data;
  Sensor = block.InputPort(2).Data;

  [delta_f, T_FR, Id_FR, T_FL, Id_FL, T_RR, Id_RR, T_RL, Id_RL] = meso_controller_core(Tall, Sensor);

  block.OutputPort(1).Data = delta_f;
  block.OutputPort(2).Data = T_FR;
  block.OutputPort(3).Data = Id_FR;
  block.OutputPort(4).Data = T_FL;
  block.OutputPort(5).Data = Id_FL;
  block.OutputPort(6).Data = T_RR;
  block.OutputPort(7).Data = Id_RR;
  block.OutputPort(8).Data = T_RL;
  block.OutputPort(9).Data = Id_RL;

% =========================================================================
% Terminate
% =========================================================================
function Terminate(block)
  % No cleanup required
