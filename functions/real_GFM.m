function [W] = real_GFM(prm,phi,WW)
%- Purpose: Constructs one real domain from two fluid domains as per
%--- ghost-fluid method system of liquid-gas
%- Variables:
%--- phi = Level set function, defines interface fluids
%--- W = [rho rho*u (rho*v) E]^T Array of conserved variables

[n_dim,n_var,n] = deal(prm{1:3});

switch n_dim
    case 1             %===== 1D Case =====% 
        W = zeros(n_var,n);
        for i = 1:n
           if (phi(i) > 0)              % Fluid 1 regions
               W(:,i) = WW(1,:,i);
           elseif (phi(i) <= 0)         % Fluid 2 regions
               W(:,i) = WW(2,:,i);
           end
        end
        
    case 2             %===== 2D Case =====% 
        W = zeros(n_var,n(1),n(2));
        for i = 1:n(1)
            for j = 1:n(2)
               if (phi(i,j) > 0)
                   W(:,i,j) = WW(1,:,i,j);
               elseif (phi(i,j) <= 0) 
                   W(:,i,j) = WW(2,:,i,j);
               end
            end
        end
end