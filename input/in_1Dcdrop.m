ICs_hdr = "%------------------ 1D Option - Centered Liquid Droplet -----------------%";
%- Regions:
%--- (1) Gas, (2) Liquid
%- Dimensions:
%--- d(1) = Length of liquid droplet
%--- d(2) = x-coordinate of liquid droplet center
%- Schematic:
%                   x_min                         x_max
%                    |--- 1 ---==== 2 ====--- 1 ---|
%                              |<- d(1) ->|   
%                    |<--- d(2) --->|   

                       %===== Parameters =====%
n_dim = 1;                              % -> # of spatial dimensions
dx = 0.005;                              % -> Grid spacing [m]
x_min = 0; x_max = 1;                   % -> Grid boundaries [m]
d = [0.2,0.5];                          % -> Dimensions [m]
U_r(1,:) = [1.226,0,1.0e5];             % -> Region (1) variables
U_r(2,:) = [1000,100,1.0e5];           % -> Region (2) variables
flg_fld = [0,1];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","perfect"];            % -> Pick EoS
c_EoS = {1.4,1.4};                      % -> EoS parameters
slvr = ["roe_perfect","incomp_1D"];     % -> Pick solver
cfl = 0.1;                              % -> CFL # to get time step
t_f = 7.5e-4;                           % -> Final simulation time [s]
flg_BCs = 0;                            % -> Boundary conditions option 
                                        
                      %===== Output Options =====%   
n_out = 76;                             % -> # of output grid points
wrt_nm = 'flow_1Dcdrop';                % -> File name for data writing
opt_plt = 1;                            % -> Plot 1D results?
t_anmt = 0.1;                           % -> Filler time between plot animation [s]                  
n_anmt = 5;                             % -> # of solver iterations per plot animation

                   %===== Intrinsics (Don't Change) =====% 
grid_setup;                             % Build grid before assigning ICs
for i = 1:n                             % Assign initial properties
    if (x(i) >= d(2) - d(1)/2 && x(i) <= d(2) + d(1)/2)
        U(:,i) = U_r(2,:);
    else
        U(:,i) = U_r(1,:);
    end
end
for i = 1:n                             % Build initial level set
   if (x(i) <= d(2))
       phi(i) = d(2) - d(1)/2 - x(i);
   else
       phi(i) = x(i) - d(2) - d(1)/2;
   end
end
