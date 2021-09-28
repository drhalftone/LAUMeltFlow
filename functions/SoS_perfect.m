function a = SoS_perfect(n_dim,gam,U)
%- Purpose: Computes speed of sound 'a' for a perfect gas

switch n_dim
    case 1             %===== 1D Case =====% 
        a = sqrt(gam*U(3)/U(1));

    case 2             %===== 2D Case =====%
        a = sqrt(gam*U(4)/U(1)); 
end