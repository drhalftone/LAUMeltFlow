function U = prim_perfect(n_dim,gam,W)
%- Purpose: Computes conserved variables 'W' for a perfect gas

switch n_dim
    case 1             %===== 1D Case =====% 
        U(1) = W(1);                    % u_1 = rho
        U(2) = W(2)/W(1);               % u_2 = rho*u/rho
        U(3) = (gam-1)*(W(3) - 1/2*W(1)*(W(2)/W(1))^2);       
                                        % u_3 = p

    case 2             %===== 2D Case =====%
        U(1) = W(1);                    % u_1 = rho
        U(2) = W(2)/W(1);               % u_2 = rho*u/rho
        U(3) = W(3)/W(1);               % u_3 = rho*v/rho
        U(4) = (gam-1)*(W(4) - 1/2*W(1)*((W(2)/W(1))^2 + (W(3)/W(1))^2));       
                                        % u_4 = p    
end