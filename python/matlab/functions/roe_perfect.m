function W = roe_perfect(prm,fld,dt,X,phi,U,W)
%- Purpose: Advance one time step of compressible fluid in conserved variables W 
%- Method: Roe's Approximate 1st-order. Updates all grid points with 
%--- dimensional splitting when n_dim > 1.
%- Assumptions: 
%--- Grid = Structured, rectangular 
%- Variables:
%--- W = [rho rho*u (rho*v) E]^T Array of conserved variables

[n_dim,n_var,n,dx,c_EoS,~,flg_BCs,n_nds] = deal(prm{1:4},prm{11:14});
gam = cell2mat(c_EoS(fld));                 % Pass specific heat ratio


if (n_dim == 1)              %===== 1D Case =====%
    F_iph = zeros(n_var,n);                 % Allocate flux vector
    for i = 1:n                             % Calculate flux
        i_l = i; i_r = i+1;
        if (i == n)                         % Compute i=n flux for periodic condition
            i_r = 1;
        end
       F_iph(:,i) = roe_flux1D(n_dim,gam,W(:,i_l),W(:,i_r));
    end
    for i = 1:n
       i_l = i-1; i_r = i;                  % Boundary conditions
       if (i == 1)                          % Left end
            switch flg_BCs(1)
                case 0                      % Dirichlet 
                    i_l = 1; i_r = 1;
                case 1                      % Neumann 
                    i_l = 2; i_r = 1;
                case 2                      % Periodic
            end
       end
       if (i == n)                          % Right end
            switch flg_BCs(2)
                case 0                      % Dirichlet
                    i_l = 1; i_r = 1;
                case 1                      % Neumann
                    i_l = n-1; i_r = n-2;
                case 2                      % Periodic
                    i_l = n; i_r = 1;
            end
       end
       for k = 1:n_var                      % Update step
           W(k,i) = W(k,i) - dt/dx*(F_iph(k,i_r)-F_iph(k,i_l));
       end
    end
        
elseif (n_dim == 2)     %===== 2D Case =====%  

                          %--- x-Sweep ---%
    F_iph = zeros(n_var,n(1),n(2));
    parfor (j = 1:n(2),n_nds)
        F_slc = zeros(n_var,n(1));          % Allocate flux vector  
        W_slc = W(:,:,j);                   % Slice in y-direction
        for i = 1:n(1)                      % Calculate x-flux F_(i+1/2,x)
            i_l = i; i_r = i+1;
            if (i == n(1)),i_r = 1; end     % Compute i=n flux for periodic condition  
            F_slc(:,i) = roe_flux2D(n_dim,2,gam,W_slc(:,i_l,1),W_slc(:,i_r,1));
        end                                 % Boundary conditions
       for i = 1:n(1)
            i_l = i-1; i_r = i;
            if (i == 1)                     % Left edge
                switch flg_BCs(1)
                    case 0                  % Dirichlet 
                        i_l = 1; i_r = 1;
                    case 1                  % Neumann 
                        i_l = 2; i_r = 1;
                    case 2                  % Periodic
                end
            end
            if (i == n(1))                  % Right edge
                switch flg_BCs(3)
                    case 0                  % Dirichlet 
                        i_l = 1; i_r = 1;
                    case 1                  % Neumann 
                        i_l = i-1; i_r = i-2;
                    case 2                  % Periodic
                        i_l = i; i_r = 1;
                end            
            end
            for k = 1:n_var                 % Update variables
                W_slc(k,i,1) = W_slc(k,i,1) - dt/dx(2)*(F_slc(k,i_r)-F_slc(k,i_l));
            end
       end
        W(:,:,j) = W_slc;                   % Pass updated slice
    end

                            %--- y-Sweep ---%
    F_iph = zeros(n_var,n(1),n(2));
    parfor (i = 1:n(1),n_nds)
        F_slc = zeros(n_var,n(2));          % Allocate flux vector 
        W_slc = W(:,i,:);
        for j = 1:n(2)                      % Calculate y-flux F_(i+1/2,y)
            j_l = j; j_r = j+1;
            if (j == n(2)),j_r = 1; end     % Compute j=n flux for periodic condition  
            F_slc(:,j) = roe_flux2D(n_dim,1,gam,W_slc(:,1,j_l),W_slc(:,1,j_r));               
        end
        for j = 1:n(2)
            j_l = j-1; j_r = j;
            if (j == n(2))                  % Top edge
                switch flg_BCs(2)
                    case 0                  % Dirichlet 
                        j_l = 1; j_r = 1;
                    case 1                  % Neumann 
                        j_l = j-1; j_r = j-2;
                    case 2                  % Periodic
                        j_l = j; j_r = 1;
                end            
            end
            if (j == 1)                     % Bottom edge
                switch flg_BCs(4)
                    case 0                  % Dirichlet 
                        j_l = 1; j_r = 1;
                    case 1                  % Neumann 
                        j_l = 2; j_r = 2;
                    case 2                  % Periodic
                end            
            end        
            for k = 1:n_var
                W_slc(k,1,j) = W_slc(k,1,j) - dt/dx(1)*(F_slc(k,j_r)-F_slc(k,j_l));
            end
        end
        W(:,i,:) = W_slc;
    end
end