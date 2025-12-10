function a = entrp_perfect(var,n_dim,gam,U,s)
%- Purpose: Computes or inverts entropy 's' for a perfect gas
%- Variables:
%--- var = 0: Compute entropy 's' from primitive variables 'U'
%--- var = 1: Compute density 'rho = U(1)' from entropy 's' and other primitive variables

c = 0;

switch n_dim
    case 1             %===== 1D Case =====% 
        switch var
            case 0                      % Compute entropy
                a = log(U(3)) - gam*log(U(1)) + c;
            case 1                      % Compute density
                a = exp((log(U(3)) - s + c)/gam);
        end

    case 2             %===== 2D Case =====%
        switch var
            case 0
                a = log(U(4)) - gam*log(U(1));
            case 1
                a = exp((log(U(4)) - s)/gam);
        end
end

end

