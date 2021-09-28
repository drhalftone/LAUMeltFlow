ICs_hdr = "%-------- 1D Test Case - Single Fluid Compressible w/Center Zone --------%";
%- Notes: Test case with same regions/dimensions as 1D center drop, but purely gas. 

                       %===== Parameters =====%
n_dim = 1;                              % -> # of spatial dimensions
dx = 0.02;                              % -> Grid spacing
x_min = 0; x_max = 1;                   % -> Grid boundaries [m]
d = [0.2,0.5];                          % -> Dimensions
U_r(1,:) = [1.226,0,1.0e5];             % -> Region (1) variables
U_r(2,:) = [1.226,100,1.0e5];           % -> Region (2) variables
flg_fld = [0,1];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","none"];               % -> Pick EoS
c_EoS = {1.4,1};                        % -> EoS parameters
slvr = ["roe_perfect","none"];          % -> Pick solver
cfl = 0.9;                              % -> CFL # to get time step
t_f = 2.5e-4;                           % -> Final simulation time [s]
flg_BCs = 0;                            % -> Boundary conditions option 
                                        
                      %===== Output Options =====%   
n_out = 55;                             % -> # of output grid points 
wrt_nm = 'flow_1Dcomp1fl';              % -> File name for data writing
opt_plt = 1;                            % -> Plot 1D data?
t_anmt = 0.7;                           % -> Filler time between plot animation [s]                  
n_anmt = 2;                             % -> # of solver iterations per plot animation

                   %===== Intrinsics (Don't Change) =====% 
grid_setup;                             % Build grid before assigning ICs
for i = 1:n                             % Assign initial properties
    if (x(i) >= d(2) - d(1)/2 && x(i) <= d(2) + d(1)/2)
        U(:,i) = U_r(2,:);
    else
        U(:,i) = U_r(1,:);
    end
end
phi(1:n) = 1;                           % Make level set pure gas