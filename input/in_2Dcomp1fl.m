ICs_hdr = "%------------- 2D Option - Pure Compressible w/Center Zone --------------%";
%- Notes: Test case with same regions/dimensions as 2D center drop, purely gas.  

n_dim = 2;                              % -> # of spatial dimensions
dx = 0.02;                              % -> x and y-grid spacings
x_min = [0,0]; x_max = [2,1];           % -> Grid boundaries
d = [0.20,0.5,0.5];                     % -> Dimensions
U_r(1,:) = [1.226,0,0,1.0e5];           % -> Region (1) variables
U_r(2,:) = [1.226,600,0,1.0e5];         % -> Region (2) variables
    
                       %===== Parameters =====%
flg_fld = [0,1];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","perfect"];            % -> Pick EoS
c_EoS = {1.4,1.4};                      % -> EoS parameters
slvr = ["roe_perfect","none"];          % -> Pick solver
cfl = 0.9;                              % -> CFL # to get time step
t_f = 2.5e-4;                           % -> Final simulation time [s]
flg_BCs = 0;                            % -> Boundary conditions option 
n_nds = 0;                              % -> # of parallel nodes for 
                                            % iterations (0=serial,inf=maximum)
                                        
                      %===== Output Options =====%   
n_disp = 5;                             % -> Command window display counter
dx_out = 0.04;                          % -> # of output grid points 
wrt_nm = 'flow_2Dtstcomp';              % -> File name for data writing
opt_plt = 1;                            % -> 2D Plotting option 
t_anmt = 0.1;                           % -> Filler time between plot animation [s]                  
n_anmt = 1;                             % -> # of solver iterations per plot animation

                   %===== Intrinsics (Don't Change) =====% 
grid_setup;                             % Build grid before assigning ICs
for i = 1:n(1)                          % Assign properties
    for j = 1:n(2)
        if (norm([x(1,i)-d(2),x(2,j)-d(3)]) <= d(1)/2)
            U(:,i,j) = U_r(2,:);
        else
            U(:,i,j) = U_r(1,:);
        end
    end
end
phi(1:n(1),1:n(2)) = 1;                 % Make level set pure gas