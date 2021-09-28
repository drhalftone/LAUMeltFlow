function F_iph = roe_flux1D(n_dim,gam,W_L,W_R)
%- Purpose: Calculates flux F_(i+1/2) for x-sweep of Roe's approximate
%--- solver between two points of a Riemann problem
%- Variables:
%--- gam = Specific heat ratio
%--- [W_L,W_R] = Conserved variables to left and right of Riemann problem

n_var = n_dim + 2;                      % # of primitive/conserved variables

                      %===== Pre-Processing =====%
rho_L = W_L(1); rho_R = W_R(1);         % Densities - left and right
u_L = W_L(2)/W_L(1);                    % Velocities - left and right
u_R = W_R(2)/W_R(1);
p_L = pres(n_dim,gam,W_L);              % Pressure - left and right
p_R = pres(n_dim,gam,W_R);
E_L = W_L(3); E_R = W_R(3);             % Total energy - left and right
h_L = (E_L + p_L)/rho_L;                % Enthalpy - lefta and right
h_R = (E_R + p_R)/rho_R;
drho = rho_R - rho_L;                   % Primitive variable differences
du = u_R - u_L;
dp = p_R - p_L;
%dW = W_R - W_L;                         % Difference in conserved variables

                  %===== Left and Right-side Flux =====%  
F_L(1) = rho_L*u_L;
F_L(2) = rho_L*u_L^2 + p_L;
F_L(3) = u_L*(E_L+p_L);
F_R(1) = rho_R*u_R;
F_R(2) = rho_R*u_R^2 + p_R;
F_R(3) = u_R*(E_R+p_R);

                       %===== Roe-Averages =====%
rho = sqrt(rho_R*rho_L);                % Roe-averaged density
u = roe_avg(rho_L,rho_R,u_L,u_R);       % Roe-averaged velocities 
h = roe_avg(rho_L,rho_R,h_L,h_R);       % Roe-averaged enthalpy
a = sqrt((gam-1)*(h-1/2*u^2));          % Roe-averaged speed of sound   

                     %===== Eigenvalues/vectors =====%
lmda = [u;u+a;u-a];                     % Eigenvalues
dv = [drho - dp/a^2;du + dp/(rho*a); ...% Wave strengths 
    du - dp/(rho*a)];
r(1,:) = [1;u;1/2*u^2];                 % Eigenvectors
r(2,:) = rho/(2*a)*[1;u+a;h+a*u];  
r(3,:) = -rho/(2*a)*[1;u-a;h-a*u];  

                          %===== Flux =====%
%F_iph = 1/2*(F_L + F_R) - 1/2*(abs(u) + a)*dW'; % Rusanov flux - test case
flux_sum = zeros(1,n_var);
for j = 1:n_var
    flux_sum = flux_sum + dv(j)*abs(lmda(j))*r(j,:);
end
F_iph = 1/2*(F_L + F_R) - 1/2*flux_sum; % Roe flux


function r_roe = roe_avg(rho_L,rho_R,r_L,r_R)
%- Purpose: Calculates the Roe-averaged value of a property 'r'
%--- based on values on the left and right sides of a Riemann problem
%- Variables:
%--- [rho_L,rho_R] = Densities on left and right sides
r_roe = (sqrt(rho_L)*r_L + sqrt(rho_R)*r_R)/(sqrt(rho_L) + sqrt(rho_R));
end      
end