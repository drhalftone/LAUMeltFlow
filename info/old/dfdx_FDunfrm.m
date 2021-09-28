function dfdx = dfdx_FDunfrm(n_dim,k,dx,F)
%- Purpose: Computes the derivative 'dfdx' of a function 'f' on grid 'x'. 
%--- Uses centered differences in center points, backward and forward otherwise.
%- Variables:
%--- n_dim = # of dimensions on grid
%--- k = Order of accuracy

switch n_dim
    case 1
        npt = length(F);
        dfdx = zeros(1,npt);
        switch k
            case 2
                for i = 1:npt
                    if (i == 1) % Forward difference (2nd order)
                        dfdx(i) = 1/dx*(-3/2*F(i) + 2*F(i+1) - 1/2*F(i+2));
                    elseif (i >= 2 && i <= npt-1) % Centered difference (2nd order)
                        dfdx(i) = 1/(2*dx)*(F(i+1) - F(i-1));
                    else %(i == npt) % Backward difference (2nd order)
                        dfdx(i) = 1/dx*(3/2*F(i) - 2*F(i-1) + 1/2*F(i-2));
                    end
                end
        end
end