function u_n = godunov(n_dim,lmda,u,f)
%- Purpose: Updates a time step for a grid point using a stencil of the
%--- function 'u' and flux 'f' using k-th order Godunov's upwind method

if (n_dim == 1)         %===== 1D Case =====%
    e = zeros(1,2);
    for l = 1:2
        if (u(l+1) == u(l))             % Stencil: [i-1,i,i+1] = [1,2,3]
            e(l) = 0;                   % Compute artificial viscosity
        else
            e(l) = max([(f(l+1)-f(l))/(u(l+1)-u(l)),(-f(l+1)+f(l))/(u(l+1)-u(l))]);
        end
    end                                 % Compute u^(n+1)
    u_n = u(2) - lmda/2*(f(3)-f(1)) ...
        + lmda/2*(e(2)*(u(3)-u(2)) - e(1)*(u(2)-u(1)));
elseif (n_dim == 2)     %===== 2D Case =====%
    e = zeros(n_dim,2);
    for dim = 1:n_dim
        for l = 1:2
            if (u(dim,l+1) == u(dim,l)) % Stencil: [i-1,i,i+1] = [1,2,3]
                e(dim,l) = 0;           % Compute artificial viscosity
            else
                e(dim,l) = max([(f(dim,l+1)-f(dim,l))/(u(dim,l+1)-u(dim,l)),(-f(dim,l+1)+f(dim,l))/(u(dim,l+1)-u(dim,l))]);
            end
        end                             % Compute u^(n+1)         
    end
    u_n = u(1,2) - lmda(1)/2*(f(1,3)-f(1,1)) ...
        + lmda(1)/2*(e(1,2)*(u(1,3)-u(1,2)) - e(1,1)*(u(1,2)-u(1,1))) ...
    - lmda(2)/2*(f(2,3)-f(2,1)) ...
        + lmda(2)/2*(e(2,2)*(u(2,3)-u(2,2)) - e(2,1)*(u(2,2)-u(2,1)));        
end