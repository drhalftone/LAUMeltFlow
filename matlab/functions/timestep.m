function dt = timestep(prm,U,a)
%- Purpose: Computes time step 'dt' from 2D Euler primitive variables 'U'
%- Method: Uses CFL number with maximum grid point velocity magnitude
%- Variables:
%--- U = [rho u (v) E]^T, Array of primitive variables
%--- a = sqrt(gam*p/rho), Speed of sound
%--- dt = CFL*dx/(|V|_max + a)
[n_dim,~,n,dx,~,~,~,cfl] = deal(prm{1:8});

switch n_dim
    case 1             %===== 1D Case =====%   
        dt = 1/2*cfl*dx/max(abs(U(2,:)) + a);
        
    case 2             %===== 2D Case =====% 
        V_norm = zeros(n(1),n(2));      % Allocate arrays on grid
        for i = 1:n(1)%,n_nds)
            for j = 1:n(2)
                V_norm(i,j) = norm([U(2,i,j),U(3,i,j)]);
            end
        end
        dt = 1/2*cfl*min(dx)/max(abs(V_norm) + a,[],'all');
end