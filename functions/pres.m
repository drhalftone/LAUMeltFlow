function p = pres(n_dim,gam,W)
%- Purpose: Calculate pressure from conserved variables 'W'
%- Assumption: Perfect gamma-law gas relation in pressure 'p' 
%--- to total energy 'E' 
%- Equation: p = (gam-1)*(E - 1/2*rho*(u^2 + v^2))
%- Variables:
%--- W = [rho rho*u (rho*v) E]^T Array of conserved variables

if (n_dim == 1)           %===== 1D Case =====% 
    p = (gam-1)*(W(3) - 1/2*W(1)*(W(2)/W(1))^2);       
    
elseif (n_dim == 2)       %===== 2D Case =====% 
    p = (gam-1)*(W(4) - 1/2*W(1)*((W(2)/W(1))^2 + (W(3)/W(1))^2));
end

