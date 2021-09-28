function I = advc(prm,dt,V,I)
%- Purpose: Advect a scalar function 'M' timesteps over grid
%- Equation: dI/dt + V*grad(I) = 0
%- Variables:
%--- I = Scalar function
%--- V = [u (v)] Flow velocity
%--- U = [rho u (v) P]^T Array of primitive variables

[n_dim,~,n,dx] = deal(prm{1:4});

if (n_dim == 1)            %===== 1D Case =====% 
    f = V.*I;                           % Advective flux
    In = I;                             % Store I^n
    lmda = dt/dx;
    for i = 1:n
        I_i = i-1:i+1; f_i = i-1:i+1;
        if (i == 1)
            I_i = [i,i,i]; f_i = [i,i+1,i+2];
        elseif (i == n)
            I_i = [i,i,i]; f_i = [i-2,i-1,i];
        end
        I(i) = godunov(n_dim,lmda,In(I_i),f(f_i));
    end
elseif (n_dim == 2)        %===== 2D Case =====% 
    u = squeeze(V(1,:,:)); v = squeeze(V(2,:,:)); % Velocity components
    In = I;                             % Store I^n       
    lmda = dt./dx;
    for i = 1:n(1)
        for j = 1:n(2)
            I_i = i-1:i+1; f_i = i-1:i+1; 
            I_j = j-1:j+1; f_j = j-1:j+1;
            if (i > 1 && i < n(1) && j > 1 && j< n(2))
            else
                if (i == 1)
                    I_i = [i,i,i]; f_i = [i,i+1,i+2]; 
                end
                if (i == n(1))
                    I_i = [i,i,i]; f_i = [i-2,i-1,i];                         
                end
                if (j == 1)
                    I_j = [j,j,j]; f_j = [j,j+1,j+2];                           
                end
                if (j == n(2))
                    I_j = [j,j,j]; f_j = [j-2,j-1,j];                           
                end
            end
            Iv(1,:) = In(I_i,j); 
            f(1,:) = u(f_i,j).*In(f_i,j);
            Iv(2,:) = In(i,I_j); f(2,:) = v(i,f_j).*In(i,f_j);
            I(i,j) = godunov(n_dim,lmda,Iv,f);
        end
    end
end