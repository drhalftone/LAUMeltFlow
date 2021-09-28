ICs_hdr = "%------ 2D Option - Circular Liquid Droplet with Shock Impingement ------%";
%- Regions:
%--- (1) Gas, (2) Liquid (3) Post-shocked Gas
%- Dimensions:
%--- d(1) = Diameter of liquid droplet
%--- d(2) = x-coordinate of liquid droplet center
%--- d(3) = y-coordinate of liquid droplet center
%--- d(4) = x-coordinate of shock diaphragm
%- Schematic:
%                              |<- d(1) ->| 
%             y_max  ──────────────────────────────────────────────
%                   |---------------------------===================|
%                   |--------------===----------===================|
%                   |------------=======--------===================|
%                 _ |--- 1 ----==== 2 ====------========= 3 =======|
%                 ^ |------------=======--------===================|
%            d(3) ╵ |--------------===----------===================|
%                 v |---------------------------===================|
%           y_min ─  ──────────────────────────────────────────────
%                 x_min                           x_max
%                   |<---- d(2) --->|     
%                   |<--------- d(4) --------->|     

                       %===== Parameters =====%
n_dim = 2;                              % -> # of spatial dimensions
dx = 1;                                 % -> x and y-grid spacings [m]
x_min = [0,0]; x_max = [100,50];        % -> Grid boundaries
d = [50,50,0,80];                       % -> Dimensions [m]
U_r(1,:) = [1,0,0,1];                   % -> Region (1) variables
U_r(2,:) = [0.138,0,0,1];               % -> Region (2) variables
U_r(3,:) = [1.3764,-0.394,0,1.5698];    % -> Region (3) variables
flg_fld = [0,0];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","perfect"];            % -> Pick EoS
c_EoS = {1.4,1.67};                     % -> EoS parameters
slvr = ["roe_perfect","roe_perfect"];   % -> Pick solvercfl = 0.9;                              % -> CFL # to get time step
t_f = 2.5e1;                            % -> Final simulation time [s]
flg_BCs = [0,1,0,1];                    % -> Boundary conditions option 
%n_r = 1;                                % -> # of solver iterations per reinitialization
%e_r = 0.1;                              % -> Reinitialization convergence tolerance
n_nds = inf;                            % -> # of parallel nodes for 
                                            % iterations (0=serial,inf=maximum)
                                        
                      %===== Output Options =====%   
n_disp = 3;                             % -> Command window display counter
dx_out = 1;                             % -> # of output grid points 
wrt_nm = 'flow_2Dcdropimp';             % -> File name for data writing
opt_plt = 1;                            % -> 2D Plotting option 
t_anmt = 0.0;                           % -> Filler time between plot animation [s]                  
n_anmt = 1;                             % -> # of solver iterations per plot animation
plt_ps = [10 10 600 500];               % -> Plot position/size
n_vec = [20,10];                        % -> Velocity vector grid for plotting
plt_wn = 1;                             % -> # of plot tiles in x-direction

                   %===== Intrinsics (Don't Change) =====% 
grid_setup;                             % Build grid before assigning ICs
for i = 1:n(1)                          % Assign properties
    for j = 1:n(2)
        if (norm([x(1,i)-d(2),x(2,j)-d(3)]) <= d(1)/2)
            U(:,i,j) = U_r(2,:);
        else
            if (x(1,i) < d(4))
                U(:,i,j) = U_r(1,:);
            else
                U(:,i,j) = U_r(3,:);
            end
        end
    end
end
for i = 1:n(1)
    for j = 1:n(2)
        phi(i,j) = sqrt((x(1,i)-d(2))^2 + (x(2,j)-d(3))^2) - 1/2*d(1);
    end
end