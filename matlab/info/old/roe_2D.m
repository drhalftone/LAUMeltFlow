function W = roe_2D(npt,gam,dt,dx,dy,W)
%- Purpose: Advance one time step of compressible fluid in conserved variables W 
%- Method: Roe's Approximate solver. Updates all grid points with 
%--- dimensional splitting in '(X,Y)'
%- Assumptions: 
%--- Grid = Structured, rectangular 
%--- Boundary conditions = Fixed 
%- Variables:
%--- W = [rho rho*u rho*v e_t]^T Array of conserved variables

n_prp = 4;                              % # of primitive/conserved variables

                        %===== x-Sweep =====%
for j = 2:npt(2)-1
    for i = 2:npt(1)-1
        F_iph = ...                    % Calculate x-flux F_(i+1/2,x)
            flux_2D(1,gam,W(:,i-1,j),W(:,i,j));
        for k = 1:n_prp
            W(k,i,j) = W(k,i,j) + F_iph(k)*dt/dy;
        end
    end
end

                        %===== y-Sweep =====%
for i = 2:npt(1)-1
    for j = 2:npt(2)-1
        F_iph = ...                    % Calculate y-flux F_(i+1/2,y)
            flux_2D(2,gam,W(:,i,j-1),W(:,i,j));
        for k = 1:n_prp
            W(k,i,j) = W(k,i,j) + F_iph(k)*dt/dx;
        end
    end
end