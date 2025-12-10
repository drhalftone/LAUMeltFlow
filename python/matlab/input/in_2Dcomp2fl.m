ICs_hdr = "%--------- 2D Test Case - Two Fluid Compressible w/Center Zone ----------%";
%- Regions:
%--- (1) Gas 1, (2) Gas 2
%- Dimensions:
%--- d(1) = Diameter of center zone
%--- d(2) = x-coordinate of zone center
%--- d(3) = y-coordinate of zone center
%- Schematic:
%                              |<- d(1) ->| 
%             y_max  ──────────────────────────────
%                   |------------------------------|
%                   |--------------===------- 1 ---|
%                   |------------=======-----------|
%                 _ |----------==== 2 ====---------|
%                 ^ |------------=======-----------|
%            d(3) ╵ |--------------===-------------|
%                 v |------------------------------|
%           y_min ─  ──────────────────────────────
%                 x_min                           x_max
%                   |<---- d(2) --->|     

                       %===== Parameters =====%
n_dim = 2;                              % -> # of spatial dimensions
dx = [0.02,0.02];                       % -> x and y-grid spacings
x_min = [0,0]; x_max = [1,1];           % -> Grid boundaries
d = [0.20,0.5,0.5];                     % -> Dimensions
U_r(1,:) = [1,1,1,1];                   % -> Region (1) variables
U_r(2,:) = [0.138,1,1,1];               % -> Region (2) variables
flg_fld = [0,0];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","perfect"];            % -> Pick EoS
c_EoS = {1.4,1.67};                     % -> EoS parameters
slvr = ["roe_perfect","roe_perfect"];   % -> Pick solver
cfl = 0.9;                              % -> CFL # to get time step
t_f = 0.3;                              % -> Final simulation time [s]
flg_BCs = 0;                            % -> Boundary conditions option
%n_r = 1;                               % -> # Solver iterations per level set reinit 
%e_r = 5e-4;                            % -> Level set reinit convergence tolerance
n_nds = inf;                            % -> # of parallel nodes for 
                                            % iterations (0=serial,inf=maximum)
                                        
                      %===== Output Options =====%   
n_disp = 5;                             % -> Command window display counter
%n_out = [55,55];                        % -> # of output grid points 
flg_wrt = 1;                            % -> Write data out?
wrt_nm = 'flow_2Dcomp2fl';              % -> File name for data writing
opt_plt = 2;                            % -> 2D Plotting option (1=velocity 
t_anmt = 0.0;                           % -> Filler time between plot animation [s]                  
n_anmt = 5;                             % -> # of solver iterations per plot animation
plt_ps = [10 10 1200 700];              % -> Plot position/size
n_vec = [20,10];                        % -> Velocity vector grid for plotting
plt_wn = 3;                             % -> # of plot tiles in x-direction

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
for i = 1:n(1)
    for j = 1:n(2)
        phi(i,j) = sqrt((x(1,i)-d(2))^2 + (x(2,j)-d(3))^2) - 1/2*d(1);
    end
end