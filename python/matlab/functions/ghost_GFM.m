function [UU,WW] = ghost_GFM(prm,X,phi,U)
%- Purpose: Constructs two domains as per the ghost-fluid method for 
%--- gas-liquid systems. Applies Rankine-Hugoniot jump conditions.
%- Variables:
%--- phi = Level set function, defines interface between fluids
%--- W = [rho rho*u (rho*v) E]^T Array of conserved variables

[n_dim,n_var,n,flg_fld,~,c_EoS,flg_dbg] = deal(prm{1:3},prm{9:11},prm{39});
c_EoS = cell2mat(c_EoS);

if (n_dim == 1)           %===== 1D Case =====%
%- Method:
%--- Ghost gas: Copies velocities 'u' and pressure 'p' from real liquid,
%------ extrapolates entropy 's' from adjacent real gas.
%--- Ghost liquid: Copies velocities 'u' and pressure 'p' from real gas,
%------ extrapolates entropy 's' from adjacent real gas.
    UU = zeros(2,n_var,n);              % Allocate new domains
        s = zeros(1,n);

                        %----- Extrapolate -----%

    if (flg_fld(1) == 0)                % Fluid 1
        s = zeros(1,n);
        for i = 1:n
            if (phi(i) > 0)             % Compute entropy in gas regions
                s(i) = entrp_perfect(0,n_dim,c_EoS(1),U(:,i),s(i));
            end
        end
        s = extrp(1,prm,X,phi,s);
        for i = 1:n
            if (phi(i) <= 0)            % Invert entropy -> density
                UU(1,1,i) = entrp_perfect(1,n_dim,c_EoS(1),U(:,i),s(i));
            end
        end
    end
    if (flg_dbg)
        figure(99)
        length(X)
        length(s)
        % plot(X(phi<0),s(phi<0),'-or')
        % hold on;
        % plot(X(phi>0),s(phi>0),'-ob')
        hold off
        % for i = 1:n-1
        %    if (phi(i+1)*phi(i) < 0), plot([X(i),1;X(i),50],'-'); end
        % end
    end
    if (flg_fld(2) == 0)                % Fluid 2
        s = zeros(1,n);
        for i = 1:n     
            if (phi(i) <= 0)            % Compute entropy in gas regions
                s(i) = entrp_perfect(0,n_dim,c_EoS(2),U(:,i),s(i));
            end
        end
        s = extrp(2,prm,X,phi,s);
        for i = 1:n
            if (phi(i) > 0)             % Invert entropy -> density
                UU(2,1,i) = entrp_perfect(1,n_dim,c_EoS(2),U(:,i),s(i));
            end
        end
    end
    if (flg_dbg)
        figure(100)
        for i = 1:n-1
           if (phi(i+1)*phi(i) < 0), plot([X(i),1;X(i),50],'-'); end
        end
        % plot(X(phi<0),s(phi<0),'-or')
        % hold on;
        % plot(X(phi>0),s(phi>0),'-ob')
        hold off
    end

                            %----- Copy -----%
    for i = 1:n
        if (phi(i) > 0)                 % Fluid 1 region
            UU(1,:,i) = U(:,i);         % Copy real fluid 1 -> fluid 1
            UU(2,2:3,i) = U(2:3,i);     % Copy real fluid 2 velocity, pressure -> ghost fluid 1
        elseif (phi(i) <= 0)            % Fluid 2 region
            UU(2,:,i) = U(:,i);         % Copy real fluid 2 -> fluid 2
            UU(1,2:3,i) = U(2:3,i);     % Copy real fluid 1 velocity, pressure -> ghost fluid 2
        end
    end
    WW(1,:,:) = state_var(prm,"cons",n_var,phi,squeeze(UU(1,:,:))); 
    WW(2,:,:) = state_var(prm,"cons",n_var,phi,squeeze(UU(2,:,:))); 
    
elseif (n_dim == 2)        %===== 2D Case =====%
%- Method:
%--- Ghost gas: 
%--- Ghost liquid: 
    UU = zeros(2,n_var,n(1),n(2));      % Allocate gas and liquid variables

                        %----- Extrapolate -----%
    if (flg_fld(1) == 0)                % Fluid 1
        s = zeros(n(1),n(2));
        for i = 1:n(1)
            for j = 1:n(2)
                if (phi(i,j) > 0)       % Compute entropy in real gas 1
                    s(i,j) = entrp_perfect(0,n_dim,c_EoS(1),U(:,i,j),s(i,j));
                end
            end
        end
        s = extrp(1,prm,X,phi,s);
        for i = 1:n(1)
            for j = 1:n(2)
                if (phi(i,j) <= 0)      % Invert entropy -> density
                    UU(1,1,i,j) = entrp_perfect(1,n_dim,c_EoS(1),U(:,i,j),s(i,j));
                end
            end
        end
    end
    if (flg_fld(2) == 0)                % Fluid 2
        s = zeros(n(1),n(2));
        for i = 1:n(1)
            for j = 1:n(2)
                if (phi(i,j) <= 0)      % Compute entropy in real gas 2
                    s(i,j) = entrp_perfect(0,n_dim,c_EoS(2),U(:,i,j),s(i,j));
                end
            end
        end
        s = extrp(2,prm,X,phi,s);
        for i = 1:n(1)
            for j = 1:n(2)
                if (phi(i,j) > 0)       % Invert entropy -> density
                    UU(2,1,i,j) = entrp_perfect(1,n_dim,c_EoS(2),U(:,i,j),s(i,j));
                    %UU(2,1,i,j) = 0.138;
                end
            end
        end
    end

                           %----- Copying -----%
    for i = 1:n(1)
        for j = 1:n(2)
            if (phi(i,j) > 0)           % Gas regions
                UU(1,:,i,j) = U(:,i,j); % Copy real fluid 1 -> fluid 1
                UU(2,2:4,i,j) = U(2:4,i,j); % Copy real fluid 2 velocity, pressure -> ghost fluid 1
            elseif (phi(i,j) <= 0)      % Fluid 2 region
                UU(2,:,i,j) = U(:,i,j); % Copy real fluid 2 -> real fluid 2
                UU(1,2:4,i,j) = U(2:4,i,j); % Copy real fluid 1 velocity, pressure -> ghost fluid 2
            end
        end
    end
    WW(1,:,:,:) = state_var(prm,"cons",n_var,phi,squeeze(UU(1,:,:,:))); 
    WW(2,:,:,:) = state_var(prm,"cons",n_var,phi,squeeze(UU(2,:,:,:))); 
end 


end

