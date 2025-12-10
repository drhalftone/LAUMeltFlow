function phi_n = eikonal(n_dim,dx,dt,phi_0,phi)
%- Purpose: Discretly solve the Eikonal equation
%- Equation: dphi/dt = S(phi_0)(1-|grad(phi)|)
%- Smoothing: S(phi_0) = phi_0/sqrt(phi_0^2 + min(dx,dy))

epsln = min(dx);
S = phi_0/sqrt(phi_0^2 + epsln^2);

if (n_dim == 1)         %===== 1D Case =====%
    if (phi_0 == 0)
        G = 0;
    else
        a = (phi(2) - phi(1))/dx; b = (phi(3) - phi(2))/dx;
        G = sqrt(max(a^2,b^2)) - 1;
    end
    phi_n = phi(2) - dt*S*G;
    
elseif (n_dim == 2)       %===== 2D Case =====%
    if (phi_0 == 0)
        G = 0;
    else
        a = (phi(1,2) - phi(1,1))/dx(1); b = (phi(1,3) - phi(1,2))/dx(1);
        c = (phi(1,2) - phi(2,1))/dx(2); d = (phi(2,3) - phi(1,2))/dx(2);
        G = sqrt(max(a^2,b^2) + max(c^2,d^2)) - 1;
    end
    phi_n = phi(1,2) - dt*S*G;
end