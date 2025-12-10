function F_iph = roe_flux2D(n_dim,dim,gam,W_L,W_R)
%- Purpose: Calculates flux F_(i+1/2) for x or y-sweep of Roe's approximate
%--- solver between two points of a Riemann problem
%- Variables:
%--- dim = 1 for x-sweep, 2 for y-sweep
%--- gam = Specific heat ratio
%--- [W_L,W_R] = Conserved variables to left and right of Riemann problem

%global n_dim
n_var = n_dim + 2;                      % # of primitive/conserved variables

                      %===== Pre-Processing =====%
rho_L = W_L(1); rho_R = W_R(1);         % Densities - left and right
u_L = W_L(2)/W_L(1);                    % Velocities - left and right
u_R = W_R(2)/W_R(1);
v_L = W_L(3)/W_L(1); 
v_R = W_R(3)/W_R(1);
p_L = pres(n_dim,gam,W_L);              % Pressure - left and right
p_R = pres(n_dim,gam,W_R);
E_L = W_L(4); E_R = W_R(4);             % Total energy - left and right
h_L = (E_L + p_L)/rho_L;                % Enthalpy - lefta and right
h_R = (E_R + p_R)/rho_R;
dW = W_R - W_L;                         % Difference in conserved variables

                       %===== Roe-Averages =====%
u = roe_avg(rho_L,rho_R,u_L,u_R);       % Roe-averaged velocities
v = roe_avg(rho_L,rho_R,v_L,v_R);  
h = roe_avg(rho_L,rho_R,h_L,h_R);       % Roe-averaged enthalpy
a = roe_spd(gam,h,u,v);                 % Roe-averaged speed of sound
V_2 = u^2 + v^2;                        % Roe-averaged squared velocity

%=============================== x-Sweep ================================%
if (dim == 1)
                  %===== Left and Right-side Flux =====%
    F_L(1) = rho_L*v_L;                 F_R(1) = rho_R*v_R;
    F_L(2) = rho_L*u_L*v_L;             F_R(2) = rho_R*u_R*v_R;
    F_L(3) = rho_L*v_L^2 + p_L;         F_R(3) = rho_R*v_R^2 + p_R;
    F_L(4) = v_L*(E_L+p_L);            F_R(4) = v_R*(E_R+p_R);
    
                     %===== Eigenvalues/vectors =====%
    lmda = [v-a,v,v,v+a];               % Eigenvalues
    K(1,:) = [1,u,v-a,h-a*v];           % Eigenvectors
    K(2,:) = [1,u,v,1/2*V_2]; 
    K(3,:) = [0,1,0,u]; 
    K(4,:) = [1,u,v+a,h+v*a];
                    
                       %===== Wave Strengths =====%
    c(1) = h - a*v;                     % Define terms for convenience
    c(2) = h + a*v;
    c(3) = v - a;
    c(4) = v + a;
    c(5) = 1/2*V_2;
    c(6) = c(1)*c(4) - c(2)*c(3) + c(3)*c(5) - c(4)*c(5) - c(1)*v + c(2)*v;
    c(7) = dW(4) - dW(2)*u + u^2*dW(1);
    alph(1) = -1/c(6)*( (c(4)*c(5)-c(2)*v)*dW(1) ...
        + (c(2)-c(5))*dW(3) + (v-c(4))*c(7) );
    alph(2) = -1/c(6)*( (c(2)*c(3)-c(1)*c(4))*dW(1) ...
        + (c(1)-c(2))*dW(3) + (c(4)-c(3))*c(7) );
    alph(3) = dW(2) - u*dW(1);
    alph(4) = 1/c(6)*( (c(3)*c(5)-c(1)*v)*dW(1) ...
        + (c(1)-c(5))*dW(3) + (v-c(3))*c(7) );
end

%=============================== y-Sweep ================================%
if (dim == 2)
                  %===== Left and Right-side Flux =====%    
    F_L(1) = rho_L*u_L;            F_R(1) = rho_R*u_R;
    F_L(2) = rho_L*u_L^2 + p_L;    F_R(2) = rho_R*u_R^2 + p_R;    
    F_L(3) = rho_L*u_L*v_L;        F_R(3) = rho_R*u_R*v_R;
    F_L(4) = u_L*(E_L+p_L);        F_R(4) = u_R*(E_R+p_R);
    
                     %===== Eigenvalues/vectors =====%
    lmda = [u-a,u,u,u+a];               % Eigenvalues
    K(1,:) = [1,u-a,v,h-a*u];           % Eigenvectors
    K(2,:) = [1,u,v,1/2*V_2]; 
    K(3,:) = [0,0,1,v]; 
    K(4,:) = [1,u+a,v,h+a*u];
                    
                       %===== Wave Strengths =====%
    c(1) = h - a*u;                     % Define terms for convenience
    c(2) = h + a*u;
    c(3) = u - a;
    c(4) = u + a;
    c(5) = 1/2*V_2;
    c(6) = c(1)*c(4) - c(2)*c(3) + c(3)*c(5) - c(4)*c(5) - c(1)*u + c(2)*u;
    c(7) = dW(4) - dW(3)*v + v^2*dW(1);
    alph(1) = -1/c(6)*( (c(4)*c(5)-c(2)*u)*dW(1) ...
        + (c(2)-c(5))*dW(2) + (u-c(4))*c(7) );
    alph(2) = -1/c(6)*( (c(2)*c(3)-c(1)*c(4))*dW(1) ...
        + (c(1)-c(2))*dW(2) + (c(4)-c(3))*c(7) );
    alph(3) = dW(3) - v*dW(1);
    alph(4) = 1/c(6)*( (c(3)*c(5)-c(1)*u)*dW(1) ...
        + (c(1)-c(5))*dW(2) + (u-c(3))*c(7) );
end

                          %===== Flux =====%
flux_sum = zeros(1,n_var);
for j = 1:n_var
    flux_sum = flux_sum + alph(j)*abs(lmda(j))*K(j,:);
end
F_iph = 1/2*(F_L + F_R) - 1/2*flux_sum; % Roe flux                          

%F_iph = 1/2*(F_L + F_R) - 1/2*(sqrt(V_2) + a)*dW'; % Rusanov flux - test case

    function r_roe = roe_avg(rho_L,rho_R,r_L,r_R)
    %- Purpose: Calculates the Roe-averaged value of a property 'r'
    %--- based on values on the left and right sides of a Riemann problem
    %- Variables:
    %--- [rho_L,rho_R] = Densities on left and right sides
    r_roe = (sqrt(rho_L)*r_L + sqrt(rho_R)*r_R)/(sqrt(rho_L) + sqrt(rho_R));
    end
    function a_roe = roe_spd(gam,h_roe,u_roe,v_roe)
    %- Purpose: Calculates the Roe-averaged value of the speed of sound
    %--- 'a' based on values on the left and right sides of a Riemann problem
    %- Variables:
    %--- gam = Specific heat ratio
    %--- [h_roe,u_roe,v_roe] = Roe-averaged enthalpy, x and y-velocity
     a_roe = sqrt((gam-1)*(h_roe - 1/2*(u_roe^2 + v_roe^2)));
    end
end