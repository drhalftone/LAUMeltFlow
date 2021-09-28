function U = ICs_2D(npt,X,Y)
%- Purpose: Generate initial conditions on 2D grid (X,Y) to call for 
%--- a CFD solver
%- Method: Assigns properties to primitive variables array 'U'. First assigns
%--- ambient values to entire field, then modifies regions depending on
%--- option variable 'opt'
%- Variables:
%--- U = [rho u v p]^T Array of primitive variables

opt = 1;                                % -> Initial condition option
n_prp = 4;                              % Number of primitive/conserved variables

                     %===== Ambient Conditions =====%
rho_a = 1.225;                          % -> Ambient density [kg/m^3]
u_a = 0;                                % -> Ambient x-velocity [m/s]
v_a = 0;                                % -> Ambient y-velocity [m/s]
P_a = 1.01325e0;                        % -> Ambient pressure [Pa]
U = zeros(n_prp,npt(1),npt(2));         % Allocate 'U'
for i = 1:npt(1)                        % Assign ambient properties
    for j = 1:npt(2)
        U(:,i,j) = [rho_a,u_a,v_a,P_a];   
    end
end

              %===== Option 1 - Constant Circular Region =====%
if (opt == 1) 
    x_c = 0.5; y_c = 0.5;               % -> Region center coordinates [m]
    d_c = 0.2;                          % -> Region diameter [m]
    rho_c = 1.225;                      % -> Region density [kg/m^3]
    u_c = 1;                            % -> Region x-velocity [m/s]
    v_c = 0;                            % -> Region y-velocity [m/s]
    P_c = 1.01325e0;                    % -> Region pressure [Pa]
    U_c = [rho_c,u_c,v_c,P_c];          % Assign region properties
    for i = 1:npt(1)
        for j = 1:npt(2)
            %[x_pnt,y_pnt] = [X(i)-r_c(1),r(2,j)-r_c(2)];
            if (norm([X(i,j)-x_c,Y(i,j)-y_c]) <= d_c/2)
                U(:,i,j) = U_c; 
            end
        end
    end
end

