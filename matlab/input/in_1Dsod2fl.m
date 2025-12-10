ICs_hdr = "%----- 1D Test Case - Two Fluid Pure Compressible Sod's Shock Tube ------%";
%- Regions:
%--- (1) Left, (2) Right
%- Dimensions:
%--- d(1) = Initial diaphragm location
%- Schematic:
%                   x_min                         x_max
%                    |----- 1 ------====== 2 ======|
%                    |<--- d(1) --->|   

                       %===== Parameters =====%
TL = 300;
TR = 300;
RL = 188.9;
RR = 287;
gamL = 1.289;
gamR = 1.4;
PL = 101325*3
PR = 101325
n_dim = 1;                              % -> # of spatial dimensions
dx = 0.001;                             % -> Grid spacing [m]
x_min = 0; x_max = 1;                   % -> Grid boundaries [m]
d = 0.5;                                % -> Dimensions [m]
U_r(1,:) = [PL/(RL*TL),0,PL]                 % -> Region (1) variables
U_r(2,:) = [PR/(RR*TR),0,PR]               % -> Region (2) variables
flg_fld = [0,0];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","perfect"];            % -> Pick EoS
c_EoS = {gamL,gamR};                      % -> EoS parameters
slvr = ["roe_perfect","roe_perfect"];   % -> Pick solver
cfl = 0.3;                              % -> CFL # to get time step
t_f = 5e-4;                           % -> Final simulation time [s]
flg_BCs = 0;                            % -> Boundary conditions option 
%n_r = 1;                                % -> # of solver iterations per reinitialization
%e_r = 1e-4;                             % -> Reinitialization convergence tolerance
n_nds = 0;                              % -> # of parallel nodes for 
                                            % iterations (0=serial,inf=maximum)
                                        
                      %===== Output Options =====%   
n_out = 76;                             % -> # of output grid points
wrt_nm = "matlab201pt";               % -> File name for data writing
opt_plt = 1;                            % -> Plot 1D data?
                                        %t_anmt = 0.5;                           % -> Filler time between plot animation [s]                  
                                         n_anmt = 2;                             % -> # of solver iterations per plot animation

                   %===== Intrinsics (Don't Change) =====% 
grid_setup;                             % Build grid before assigning ICs
for i = 1:n                             % Assign initial properties
    if (x(i) < d)
        U(:,i) = U_r(1,:);
    else
        U(:,i) = U_r(2,:);
    end
end
for i = 1:n
    phi(i) = d - x(i);
end
