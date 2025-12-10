function W = cons_perfect(n_dim,gam,U)
%- Purpose: Computes conserved variables 'W' for a perfect gas

switch n_dim
    case 1             %===== 1D Case =====% 
        W(1) = U(1);                    % w_1 = rho
        W(2) = U(1)*U(2);               % w_2 = rho*u
        W(3) = U(3)/(gam-1) + 1/2*U(1)*U(2)^2; % w_3 = E 
        
    case 2             %===== 2D Case =====%
        W(1) = U(1);                    % w_1 = rho
        W(2) = U(1)*U(2);               % w_2 = rho*u
        W(3) = U(1)*U(3);               % w_3 = rho*v
        W(4) = U(4)/(gam-1) + 1/2*U(1)*(U(2)^2 + U(3)^2);          
                                        % w_4 = E        
end