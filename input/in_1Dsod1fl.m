ICs_hdr = "%---- 1D Test Case - Single Fluid Pure Compressible Sod's Shock Tube ----%";
%- Regions:
%--- (1) Left, (2) Right
%- Dimensions:
%--- d(1) = Initial diaphragm location
%- Schematic:
%                   x_min                         x_max
%                    |----- 1 ------====== 2 ======|
%                    |<--- d(1) --->|   

                       %===== Parameters =====%
n_dim = 1;                              % -> # of spatial dimensions
dx = 0.01;                              % -> Grid spacing
x_min = 0; x_max = 1;                   % -> Grid boundaries
d = 0.5;                                % -> Dimensions
U_r(1,:) = [1,100,1e5];                 % -> Region (1) variables
U_r(2,:) = [0.125,0,1e4];               % -> Region (2) variables
flg_fld = [0,1];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","none"];               % -> Pick EoS
c_EoS = {1.4,1};                        % -> EoS parameters
slvr = ["roe_perfect","none"];          % -> Pick solver
cfl = 0.9;                              % -> CFL # to get time step
t_f = 7.5e-4;                           % -> Final simulation time [s]
flg_BCs = 1;                            % -> Boundary conditions option 
                                        
                      %===== Output Options =====%   
n_out = 51;                             % -> # of output grid points 
wrt_nm = "flow_1Dsod1fl";               % -> File name for data writing
opt_plt = 1;                            % -> Plot 1D data?
t_anmt = 1.2;                           % -> Filler time between plot animation [s]                  
n_anmt = 2;                             % -> # of solver iterations per plot animation
                                            
                   %===== Intrinsics (Don't Change) =====% 
grid_setup;                             % Build grid before assigning ICs
for i = 1:n                             % Assign initial properties
    if (x(i) <= d)
        U(:,i) = U_r(1,:);
    else
        U(:,i) = U_r(2,:);
    end
end
phi(1:n) = 1;                           % Make level set pure gas