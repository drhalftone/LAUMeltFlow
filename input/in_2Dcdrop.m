ICs_hdr = "%------------------ 2D Option - Circular Liquid Droplet -----------------%";
%- Regions:
%--- (1) Gas, (2) Liquid
%- Dimensions:
%--- d(1) = Diameter of liquid droplet
%--- d(2) = x-coordinate of liquid droplet center
%--- d(3) = y-coordinate of liquid droplet center
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

n_dim = 2;                              % -> # of spatial dimensions
dx = [0.02,0.02];                       % -> x and y-grid spacings [m]
x_min(1) = 0; x_min(2) = 0;
x_max(1) = 1; x_max(2) = 1;
% x(1,:) = 0:dx(1):1;                     % -> x-grid boundaries [m]
% x(2,:) = 0:dx(2):1;                     % -> y-grid boundaries [m]
d = [0.3125,0.3125,0.375];                      % -> Dimensions [m]
U_r(1,:) = [1.226,100,100,1.0e5];           % -> Region (1) variables
U_r(2,:) = [0.164,100,100,1.0e5];         % -> Region (2) variables
flg_fld = [0,0];
EoS = ["perfect","perfect"];            % -> Pick EoS
c_EoS = {1.4,1.667};                      % -> EoS parameters
slvr = ["roe_perfect","roe_perfect"];
                       %===== Parameters =====%
% gam = 1.4;                              % -> Specific heat ratio
cfl = 0.9;                              % -> CFL # to get time step
t_f = 3.0e-3;                           % -> Final simulation time [s]
flg_BCs = 0;                            % -> Boundary conditions option 
n_nds = 0;                            % -> # of parallel nodes for
                                            % iterations (0=serial,inf=maximum)
                                        
                      %===== Output Options =====%   
n_disp = 10;                            % -> Command window display counter
% n_out = [51,51];                        % -> # of output grid points
wrt_nm = 'flow_2Dcdrop';                % -> File name for data writing
% flg_intrp = 1;
flg_plt = 1;
opt_plt = 2;                            % -> 2D Plotting option
t_anmt = 0;

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
% phi = smth(5,n_dim,dx,x,phi);           % Smooth level set
