function W = roe(n_nds,npt,gam,dt,dx,W)
%- Purpose: Advance one time step of compressible fluid in conserved variables W 
%- Method: Roe's Approximate solver. Updates all grid points with 
%--- dimensional splitting in '(X,Y)'
%- Assumptions: 
%--- Grid = Structured, rectangular 
%--- Boundary conditions = Fixed 
%- Variables:
%--- W = [rho rho*u (rho*v) E]^T Array of conserved variables

n_prp = 4;                              % # of primitive/conserved variables

                        %===== x-Sweep =====%
parfor j = 2:npt(2)-1
    W_slc = W(:,:,j); 
    for i = 2:npt(1)-1
        F_iph = ...                    % Calculate x-flux F_(i+1/2,x)
            roe_flux2D(1,gam,W_slc(:,i-1,1),W_slc(:,i,1));
        for k = 1:n_prp
            W_slc(k,i,1) = W_slc(k,i,1) + F_iph(k)*dt/dy;
        end
    end
    W(:,:,j) = W_slc;
end

                        %===== y-Sweep =====%
parfor i = 2:npt(1)-1
    W_slc = W(:,i,:); 
    for j = 2:npt(2)-1
        F_iph = ...                    % Calculate y-flux F_(i+1/2,y)
            roe_flux2D(2,gam,W_slc(:,1,j-1),W_slc(:,1,j));
        for k = 1:n_prp
            W_slc(k,1,j) = W_slc(k,1,j) + F_iph(k)*dt/dx;
        end
    end
    W(:,i,:) = W_slc;
end
