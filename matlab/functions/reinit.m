function [phi,it] = reinit(prm,dt,phi_0)
%- Purpose: Reinitialize a signed-distance level set function phi
%-Variables:
%--- e_r = Convergence tolerance

[n_dim,~,n,dx,e_r] = deal(prm{1:4},prm{24});
phi = phi_0; it = 0; e = pi*e_r;        % Initialize solver

if (n_dim == 1)            %===== 1D Case =====% 
while (e > e_r)
    phi_prv = phi;
    it = it + 1;
    for i = 1:n
        I = i-1:i+1;
        if (i == 1)
            I = [i,i,i+1];
        elseif (i == n)
            I = [i-1,i,i];
        end
        phi(i) = eikonal(n_dim,dx,dt,phi_0(i),phi(I));
    end    
    dphi = phi - phi_prv;
    e = max(dphi);
    if (it > 1000)
        error('reinit.m: Reinitialization convergence not reached');
    end
end
 
elseif (n_dim == 2)        %===== 2D Case =====%
    while (e > e_r)
        phi_prv = phi;
        it = it + 1;
        for i = 1:n(1)
            for j = 1:n(2)
                I_i = i-1:i+1;
                I_j = j-1:j+1;
                if (i > 1 && i < n(1) && j > 1 && j < n(2))
                else
                    if (i == 1)
                        I_i = [i,i,i+1]; 
                    end
                    if (i == n(1))
                        I_i = [i-1,i,i];                        
                    end
                    if (j == 1)
                        I_j = [j,j,j+1];                            
                    end
                    if (j == n(2))
                        I_j = [j-1,j,j];                           
                    end
                end
                phi_v(1,:) = phi(I_i,j); 
                phi_v(2,:) = phi(i,I_j);
                phi(i,j) = eikonal(n_dim,dx,dt,phi_0(i,j),phi_v);
            end
        end        
        dphi = phi - phi_prv;
        e = max(dphi,[],'all');
        if (it > 1000)
            error('reinit.m: Reinitialization convergence not reached');
        end
    end
end

