%================== Ghost Fluid Method Solver - 1D/2D ===================%
%%- File: 'main.m' - Run this file to execute solver
%- Purpose: Resolves a 1D or 2D flow field of two fluids modeled as  
%--- compressible or incompressible
%- Methods:
%--- Grid = Structured, rectangular 
%- Variables: See file 'info/variables.txt'

                       %===== Intrinsics =====%
clear all;                              % Clear stored variables
close all;                              % Clear figures
warning('off','all');                   % Turn off warnings
tic;                                    % Initialize wall time  
files;                                  % Add file paths
key_call;                               % Parameter vector key

                        %===== Options =====% 
fl_in = "in_2Dsod2fl";                  % -> Choose input file

                        %===== Grid/ICs =====%
[prm,X,U,phi] = mesh_ICs(fl_in,key);    % Load parameters, mesh and ICs     

%=========================== Pre-Processing =============================%
prm_call;                               % Call parameters to main file
global t t_wall it it_r                 % Global variables for headers
it = 0; t = 0; cntr_it = 0;             % Initialize iterations
it_r = 0; cntr_r = 0;                   % Reinitialization
cntr_a = 0;                             % Animation counter
print_term(prm,1);                      % Print first header
print_term(prm,2);                      % Print initial conditions header
print_term(prm,3);                      % Print first parameters line
print_term(prm,4);                      % Print second parameters line
if (flg_plt == 1)                       % Generate figure for plotting 
    fig = plt_setup(prm);
    if (flg_intrp)
        [X_out,U_out,phi_out] ...       % Interpolate before plotting animation
            = interp(prm,X,U,phi);
    end
    plt(fig,prm,X_out,U_out,phi_out); 
end
W = state_var(prm,"cons",n_var,phi,U);  % Calculate initial conserved variables

%============================ Iterations ================================%
if (n_r > 0), print_term(prm,5); end    % Print reinitializations header
print_term(prm,6);                      % Print iterations header
while (t < t_f)
                         %===== Solver =====%
    a = state_var(prm,"SoS",1,phi,U);   % Compute speed of sound 
    dt = timestep(prm,U,a);             % Calculate time step
    if (t + dt > t_f), dt = t_f-t; end  % Adjust overshot time
    t = t + dt;                         % Tick time
    [UU,WW] = ghost_GFM(prm,X,phi,U);   % Define real/ghost domains
    WW = run_slvr(prm,dt,X,phi,UU,WW);  % Run solvers
    V = extrp_vel(prm,X,phi,WW);        % Extrapolate velocity field
    phi = advc(prm,dt,V,phi);           % Advect level set equation
    
                    %===== Reinitialization =====%
    cntr_r = cntr_r + 1; it_r = 0;      % Tick reinitialization counter
    if (cntr_r >= n_r && n_r > 0)                  
        cntr_r = 0;                     % Reset reinitialization counter
        [phi,it_r] = reinit(prm,dt,phi); % Reinitialize 
    end
    
                         %===== Results =====%
    W = real_GFM(prm,phi,WW);           % Reassemble single real domain
    U = state_var(prm,"prim",n_var,phi,W); % Compute primitive variables
    it = it + 1;                        % Tick iteration
    cntr_it = cntr_it + 1;              % Tick iteration counter
    if (it == 1 || cntr_it >= n_disp || t >= t_f)
        cntr_it = 0;                    % Reset iteration counter
        t_wall = toc;                   % Get wall time
        print_term(prm,7);              % Print iteration variables 
    end
                     %===== Animate Results =====%
    cntr_a = cntr_a + 1;                % Tick animation counter
    if (flg_anmt == 1) && (it == 1 || cntr_a >= n_anmt) 
        cntr_a = 0;
        pause(t_anmt);                  % Slow down iterations to see animation
        if (flg_intrp)
            [X_out,U_out,phi_out] ...   % Interpolate before plotting animation
                = interp(prm,X,U,phi);
        end
        plt(fig,prm,X_out,U_out,phi_out);% Animate
    end
end
print_term(prm,8);                      % Skip line

%========================== Post-Processing =============================%

                     %===== Interpolate =====%   
if (flg_intrp)
    if (n_dim == 1) 
        fprintf('Interpolating flow field (%d points)... \n',n_out);
    elseif (n_dim == 2)
        fprintf('Interpolating flow field (%d x %d points)... \n' ...
            ,[n_out(1),n_out(2)]);        
    end
    [X_out,U_out,phi_out] = interp(prm,X,U,phi);
else
    n_out = n; X_out = X; U_out = U; phi_out = phi;
end

                     %===== Write Results =====%  
if (flg_wrt)
    wrt_prm = ""; %['_t_' num2str(t)];  % -> Insert file name parameters
    wrt_fl ...                          % Construct file name   
        = append(wrt_prfx,wrt_nm,wrt_prm,wrt_sfx); 
    wrt_data(prm,X_out,U_out,phi_out,wrt_fl);
end

                     %===== Plot Results =====%  
if (flg_plt)                            % Plot results
    disp('Plotting flow field...');
    plt(fig,prm,X_out,U_out,phi_out);
end
print_term(prm,9);                      % Print final header

