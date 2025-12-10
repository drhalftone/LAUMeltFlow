function W = incomp_1D(prm,fld,dt,X,phi,U,W)
%- Purpose: Advance one time step of incompressible fluid in conserved variables W 
%- Method: Simple linear interpolation of pressure in 1D.
%- Assumptions: 
%--- Grid = Structured, rectangular 
%- Variables:
%--- W = [rho rho*u (rho*v) E]^T Array of conserved variables

[n_var,n] = deal(prm{2:3});

for i = 2:n-1                         % Find liquid regions    
    if (phi(i+1) <= 0 && phi(i) > 0)
        i_l = i+1;                      % Pressure of left-most point
        p_l = U(3,i);
    end
    if (phi(i-1) <= 0 && phi(i) > 0)
        i_r = i-1;                      % Pressure of right-most point
        p_r = U(3,i);
        if (i_l == i_r)                 % One liquid point case
            % Do nothing
        else
            pstr_l = dt/U(1,i_l)*U(3,i_l);
            pstr_r = dt/U(1,i_r)*U(3,i_r);
            for j = i_l:i_r             % Interpolate liquid pressure
                U(3,j) = p_l + (p_r-p_l)/(i_r-i_l)*(j-i_l);
                U(2,j) = U(2,j) - (pstr_r-pstr_l)/(X(i_r)-X(i_l));
            end
        end
    end
end
U(1,:) = extrp(2,prm,X,phi,U(1,:));       % Extrapolate density/velocity to outer field
U(2,:) = extrp(2,prm,X,phi,U(2,:));
W = state_var(prm,"cons",n_var,phi,U);  % Calculate initial conserved variables
