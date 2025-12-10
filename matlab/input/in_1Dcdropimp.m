ICs_hdr = "%-------------- 1D Option - Liquid Droplet Shock Impingement ------------%";
%- Regions:
%--- (1),(3) Ambient Gas, (2) Liquid, 
%- Dimensions:
%--- d(1) = Length of liquid droplet
%--- d(2) = x-coordinate of liquid droplet center
%--- d(3) = x-coordinate of shock
%- Schematic:
%              x_min                                     x_max
%               |<- d(3) ->|
%               |--- 3 ----|-- 1 --==== 2 ====----- 1 -----|
%                                  |<- d(1) ->|   
%               |<-------- d(2) ------->|   

n_dim = 1;                              % -> # of spatial dimensions
dx = 0.005;                             % -> Grid spacing [m]
x_min = 0; x_max = 1;                   % -> Grid boundaries [m]
d = [0.2,0.5,0.1];                      % -> Dimensions [m]
U_r(1,:) = [1.583,0,98066];             % -> Region (1) variables
U_r(2,:) = [10,0,1.01e5];             % -> Region (2) variables
U_r(3,:) = [2.124,89.98,1.484e5];       % -> Region (3) variables
    
                       %===== Parameters =====%
flg_fld = [0,1];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","perfect"];            % -> Pick EoS
c_EoS = {1.4,1.4};                      % -> EoS parameters
slvr = ["roe_perfect","incomp_1D"];     % -> Pick solver
cfl = 0.1;                              % -> CFL # to get time step
t_f = 1.75e-3;                           % -> Final simulation time [s]
flg_BCs = 0;                            % -> Boundary conditions option 
                                        
                      %===== Output Options =====%   
n_out = 76;                             % -> # of output grid points
wrt_nm = 'flow_1Dcdropimp';             % -> File name for data writing
opt_plt = 3;                            % -> Plot 1D data?
n_anmt = 5;                             % -> # of solver iterations per plot animation

                   %===== Intrinsics (Don't Change) =====% 
grid_setup;                             % Build grid before assigning ICs
for i = 1:n                             % Assign initial properties
    if (x(i) >= d(2) - d(1)/2 && x(i) <= d(2) + d(1)/2)
        U(:,i) = U_r(2,:);
    elseif (x(i) <= d(3))
        U(:,i) = U_r(3,:);
    else
        U(:,i) = U_r(1,:);
    end
end
for i = 1:n                           % Build initial level set
   if (x(i) <= d(2))
       phi(i) = d(2) - d(1)/2 - x(i);
   else
       phi(i) = x(i) - d(2) - d(1)/2;
   end
end
