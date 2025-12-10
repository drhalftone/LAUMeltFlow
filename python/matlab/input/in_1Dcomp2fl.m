ICs_hdr = "%--------- 1D Test Case - Two Fluid Compressible w/Center Zone ----------%";
%- Notes: Test case with same regions/dimensions as 1D center drop, but purely gas. 

                       %===== Parameters =====%
n_dim = 1;                              % -> # of spatial dimensions
dx = 0.01;                              % -> Grid spacing
x_min = 0; x_max = 1;                   % -> Grid boundaries [m]
d = [0.2,0.5];                          % -> Dimensions
U_r(1,:) = [1,1,1.0e0];                 % -> Region (1) variables
U_r(2,:) = [0.138,1,1.0e0];             % -> Region (2) variables
flg_fld = [0,0];                        % -> Treat fluid as gas (0) or liquid (1)?
EoS = ["perfect","perfect"];            % -> Pick EoS
c_EoS = {1.4,1.67};                     % -> EoS parameters
slvr = ["roe_perfect","roe_perfect"];   % -> Pick solver
cfl = 0.9;                              % -> CFL # to get time step
t_f = 1.0e-1;                           % -> Final simulation time [s]
flg_BCs = 0;                            % -> Boundary conditions option 
n_nds = 0;                              % -> # of parallel nodes for 
                                            % iterations (0=serial,inf=maximum)
                                        
                      %===== Output Options =====%   
n_out = 55;                             % -> # of output grid points 
wrt_nm = 'flow_1Dcomp2fl';              % -> File name for data writing
opt_plt = 3;                            % -> Plot 1D data?

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