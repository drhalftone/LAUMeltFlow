function V = extrp_vel(prm,X,phi,WW)
%- Purpose: Extrapolate velocity from fluid 2 domain (phi <= 0) to fluid 1
%--- domain

[n_dim,~,n] = deal(prm{1:3});

if (n_dim == 1)          %===== 1D Case =====%
    u = zeros(1,n);
    for i = 1:n
        if (phi(i) <= 0)
            u(i) = WW(2,2,i)/WW(2,1,i);     % Velocity field
        else
            u(i) = WW(1,2,i)/WW(1,1,i);
        end
    end
    V = u;
    % V = extrp(2,prm,X,phi,u);
        
elseif (n_dim == 2)        %===== 2D Case =====%
    u = zeros(n(1),n(2)); v = u;
    for i = 1:n(1)
        for j = 1:n(2)
            if (phi(i,j) <= 0)
                u(i,j) = WW(2,2,i,j)/WW(2,1,i,j); 
                v(i,j) = WW(2,3,i,j)/WW(2,1,i,j);
            else
                u(i,j) = WW(1,2,i,j)/WW(1,1,i,j);
                v(i,j) = WW(1,3,i,j)/WW(1,1,i,j);
            end
        end
    end
    V(1,:,:) = u;
    V(2,:,:) = v;
%     V(1,:,:) = extrp(2,prm,X,phi,u);
%     V(2,:,:) = extrp(2,prm,X,phi,v);
end
