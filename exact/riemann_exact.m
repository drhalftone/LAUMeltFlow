clear all
close all

RL = 287.15;
PL = 101325;
uL = 0;
% TL = 300;

RR = 287.15;
PR = 10132.5;
uR = 0;
% TR = 300;

t = 1.38e-4;
gam = 1.4;
x = 0:0.005:1;
x_0 = 0.5;

U_L = [PL/(RL*TL),uL,PL];
U_R = [PR/(RR*TR),uR,PR];

[U,~] = riemann_exact(gam,x_0,x,t,U_L,U_R);
phi(1:length(x)) = 1;

% addpath('../');
% file = ['data/sod_exact.d'];
% [fid,msg] = fopen(file,'wt');
% fprintf(fid,'x rho u p phi \n');
% fprintf(fid,'%.4f %.4f %.4f %.4f %.4f \n', ...
%     [x;U(1,:);U(2,:);U(3,:);phi]);
% fclose(fid);

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


function [U,phi] = riemann_exact(gam,x0,x,t,UL,UR)
%- Purpose: Compute Riemann problem exact solution on grid 'x' at time 't'
%- Assumption: Perfect gamma-law gas equation of state, single gamma value
%- Variables:
%--- gam = Ratio of specific heats
%--- x0 = Initial diaphragm location
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

%===== Pre-Processing =====%
    if (length(gam) == 1)
        gamL = gam;
        gamR = gam;
    elseif (length(gam) == 2)
        gamL = gam(1);
        gamR = gam(2);
    else
        error('Specific heat ratio vector *gam* should be of length two or one.')
    end

    %===== Parameters =====%
    rhoL = UL(1);                         % Density at driver end
    uL = UL(2);                           % Velocity at driver end
    pL = UL(3);                           % Pressure at driver end
    rhoR = UR(1);                         % Density at driven end
    uR = UR(2);                           % Velocity at driven end
    pR = UR(3);                           % Pressure at driven end
    xi = min(x); xf = max(x);             % Grid dimensions
    npt = length(x);                        % # of grid points

    %===== Processing =====%
    p4 = pL; rho4 = rhoL; u4 = uL;
    p1 = pR; rho1 = rhoR; u1 = uR;
    p4op1 = p4/p1;                        % Driver/driven end pressure ratio
    a4 = sqrt(gam*p4/rho4);
    a1 = sqrt(gam*p1/rho1);

    %===== Solver =====%
    eqn = @(r) P4/P1*(1 - ((gamL-1)*a1/a4*(r-1))/sqrt(2*gamR*(2*gamR+(gamR+1)*(r-1))))^(2*gamL/(gamL-1)); % Eqn 5.6
    p2op1 = fzero(eqn,pi);

    %===== Resolve Properties =====%
    gamRatio = (gamR+1)/(gamR-1);
    p2 = p2op1*p1;
    u2 = u4 + 2*a4/(gamR-1) ...                             % Eqn 5.4
         *(1-(p4op1^(-1)*p2op1)^((gamR-1)/(2*gamR)));
    a2 = a1*sqrt(p2op1*(gamRatio+p2op1)/(1+gamRatio*p2op1));    % Eqn 3.54
    u_s = u1 + a1*sqrt((gamR+1)/(2*gamR)*(p2op1-1)+1);        % Eqn 3.56
    u3 = u2;                                              % Eqn 3.57
    p3 = p2;                                              % Eqn 3.58
    a3 = a4 + (gamL-1)/2*(u4-u3);                          % Eqn 5.2

    %===== Solution =====%
    x4 = x0 + (u4-a4)*t;
    x3 = x0 + (u3-a3)*t;
    x2 = x0 + u2*t;
    x1 = x0 + u_s*t;
    for i = 1:npt
        if (x(i) < x4)
            u(i) = u4;
            a(i) = a4;
            p(i) = p4;
            rho(i) = rho4;
        elseif (x(i) >= x4 && x(i) <= x3)
            u(i) = 2/(gamL+1)*((x(i)-x0)/t + (gamL-1)/2*u4 + a4); % Eqn 3.47
            a(i) = u(i) - (x(i)-x0)/t;                  % Eqn 3.48
            p(i) = p4*(a(i)/a4)^(2*gamL/(gamL-1));         % Eqn 3.49
            rho(i) = gamL*p(i)/a(i)^2;
        elseif (x(i) >= x3 && x(i) <= x2)
            u(i) = u3;
            a(i) = a3;
            p(i) = p3;
            rho(i) = gamL*p(i)/a(i)^2;
        elseif (x(i) >= x2 && x(i) <= x1)
            u(i) = u2;
            a(i) = a2;
            p(i) = p2;
            rho(i) = gam*p(i)/a(i)^2;
        elseif (x(i) > x1)
            u(i) = u1;
            a(i) = a1;
            p(i) = p1;
            rho(i) = rho1;
        end
    end

    %===== Post-Processing =====%
    U(1,:) = rho; U(2,:) = u; U(3,:) = p;
    plot(x,p)
end
