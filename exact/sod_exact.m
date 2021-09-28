clear all
close all

RL = 287.15;
PL = 101325;
uL = 0;
TL = 300;

RR = 287.15;
PR = 10132.5;
uR = 0;
TR = 300;

t = 1.38e-4;
gam = 1.4;
x = 0:0.005:1;
x_0 = 0.5;

U_L = [PL/(RL*TL),uL,PL];
U_R = [PR/(RR*TR),uR,PR];

[U,~] = riemann_exact(gam,x_0,x,t,U_L,U_R);
phi(1:length(x)) = 1;

addpath('../'); 
file = ['data/sod_exact.d'];
[fid,msg] = fopen(file,'wt');
fprintf(fid,'x rho u p phi \n');
fprintf(fid,'%.4f %.4f %.4f %.4f %.4f \n', ...
    [x;U(1,:);U(2,:);U(3,:);phi]);
fclose(fid);

rho = U(1,:); u = U(2,:); p = U(3,:); T = p./(RL*rho);

figure(1)
plot(x,p);
xlabel('x'); ylabel('p');

figure(2)
plot(x,u); 
xlabel('x'); ylabel('u');

figure(3)
plot(x,T); 
xlabel('x'); ylabel('T');



function [U,a] = riemann_exact(gam,x_0,x,t,U_L,U_R)
%- Purpose: Compute Riemann problem exact solution on grid 'x' at time 't'
%- Assumption: Perfect gamma-law gas equation of state, single gamma value
%- Variables:
%--- gam = Ratio of specific heats
%--- x_0 = Initial diaphragm location
%--- x = Array of grid points
%--- t = Time at which to evaluate
%--- U = [rho u P]^T Array of primitive variables (left and right sides)

%                  diaphragm
%         |       |   .     |         |
%       4 |       |   3     |    2    |    1
%       <-|     <-|   .     |->       |->
%_________|_______|___._____|_________|_________
%        x4      x3  x0    x2        x1
%          expansion       contact     shock

% Note: Cited equations and procedure are based on Laney (1998)

                      %===== Parameters =====% 
rho_L = U_L(1);                         % Density at driver end 
u_L = U_L(2);                           % Velocity at driver end 
p_L = U_L(3);                           % Pressure at driver end 
rho_R = U_R(1);                         % Density at driven end 
u_R = U_R(2);                           % Velocity at driven end 
p_R = U_R(3);                           % Pressure at driven end 
x_i = min(x); x_f = max(x);             % Grid dimensions
npt = length(x);                        % # of grid points

                      %===== Processing =====% 
p_4 = p_L; rho_4 = rho_L; u_4 = u_L;
p_1 = p_R; rho_1 = rho_R; u_1 = u_R;
gam_m = gam-1;                          % Specific heat ratio notation
gam_p = gam+1;
gam_r = gam_p/gam_m;
p4op1 = p_4/p_1;                        % Driver/driven end pressure ratio
a_4 = sqrt(gam*p_4/rho_4);
a_1 = sqrt(gam*p_1/rho_1);

                       %===== Solver =====% 
eqn = @(r_p) p4op1 - r_p*(1 + gam_m/(2*a_4)*(u_4 - u_1 ... % Eqn 5.6
    - a_1/gam*(r_p-1)/(sqrt(gam_p/2/gam*(r_p-1)+1))))^(-2*gam/gam_m);
p2op1 = fzero(eqn,pi);                            

                   %===== Resolve Properties =====% 
p_2 = p2op1*p_1;
u_2 = u_4 + 2*a_4/gam_m ...                             % Eqn 5.4
    *(1-(p4op1^(-1)*p2op1)^(gam_m/(2*gam)));             
a_2 = a_1*sqrt(p2op1*(gam_r+p2op1)/(1+gam_r*p2op1));    % Eqn 3.54
u_s = u_1 + a_1*sqrt(gam_p/(2*gam)*(p2op1-1)+1);        % Eqn 3.56
u_3 = u_2;                                              % Eqn 3.57
p_3 = p_2;                                              % Eqn 3.58
a_3 = a_4 + gam_m/2*(u_4-u_3);                          % Eqn 5.2

                      %===== Solution =====% 
x_4 = x_0 + (u_4-a_4)*t;
x_3 = x_0 + (u_3-a_3)*t;
x_2 = x_0 + u_2*t;
x_1 = x_0 + u_s*t;
for i = 1:npt
    if (x(i) < x_4)
        u(i) = u_4;
        a(i) = a_4;
        p(i) = p_4;         
        rho(i) = rho_4;
    elseif (x(i) >= x_4 && x(i) <= x_3)
        u(i) = 2/gam_p*((x(i)-x_0)/t + gam_m/2*u_4 + a_4); % Eqn 3.47
        a(i) = u(i) - (x(i)-x_0)/t;                  % Eqn 3.48
        p(i) = p_4*(a(i)/a_4)^(2*gam/gam_m);         % Eqn 3.49
        rho(i) = gam*p(i)/a(i)^2;
    elseif (x(i) >= x_3 && x(i) <= x_2)
        u(i) = u_3;
        a(i) = a_3;
        p(i) = p_3;  
        rho(i) = gam*p(i)/a(i)^2;
    elseif (x(i) >= x_2 && x(i) <= x_1)
        u(i) = u_2;
        a(i) = a_2;
        p(i) = p_2;  
        rho(i) = gam*p(i)/a(i)^2;
    elseif (x(i) > x_1)
        u(i) = u_1;
        a(i) = a_1;
        p(i) = p_1;
        rho(i) = rho_1;
    end  
end

                      %===== Post-Processing =====% 
U(1,:) = rho; U(2,:) = u; U(3,:) = p;
plot(x,p)
end