function plt(fig,prm,X,U,phi)
%- Purpose: Plot primitive variable data 'U' and level set 'phi' over grid '(X,(Y))'
%- Method: Plotting options set by 'opt_plt'
%- Variables:
%--- opt_plt = 1D flag, 2D Plotting option (1=contour with velocity vectors,2=surface)
%--- U = [rho u (v) p]^T Array of primitive variables

[n_dim,n_var,~,~,x_min,x_max,opt_plt,plt_ps,~,flg_vec,n_vec] = deal(prm{1:6},prm{22},prm{34:37});

if (n_dim == 1)           %===== 1D Case =====% 
    for k = 1:n_var
        nexttile(fig,k);
        hold on; cla;
        plot(X,squeeze(U(k,:)));
    end
    nexttile(fig,n_var+1);
    hold on; cla;
    plot(X,phi);

elseif (n_dim == 2)       %===== 2D Case =====%  
    Y = squeeze(X(2,:,:)); X = squeeze(X(1,:,:));
    if (opt_plt == 1) %----- Option 1: Contour Map -----%
        if (flg_vec > 0)
            u = squeeze(U(2,:,:)); v = squeeze(U(3,:,:));
            clr_vec = [0.75 0.75 0.75];     % -> Color of velocity arrows
            if (flg_vec == 1)
                u_vec = u; v_vec = v;
                X_vec = X; Y_vec = Y;
            elseif (flg_vec > 1)
                x_vec = linspace(x_min(1),x_max(1),n_vec(1));
                y_vec = linspace(x_min(2),x_max(2),n_vec(2));
                [X_vec,Y_vec] = meshgrid(x_vec,y_vec);
                u_vec = interp2(X',Y',u',X_vec,Y_vec)';
                v_vec = interp2(X',Y',v',X_vec,Y_vec)';
                X_vec = X_vec'; Y_vec = Y_vec';
            end
        end

        clr_lb = ["\rho","{\it p}","\phi"]; k_prp = [1,4,5];
        for j = 1:3
            nexttile(fig,j);
            hold on; cla;
            if (j == 3)
                contourf(X,Y,phi);
            else
                contourf(X,Y,squeeze(U(k_prp(j),:,:)));
            end
            clr = colorbar;
            clr.Label.String = clr_lb(j);
            if (flg_vec > 0)
                quiver(X_vec,Y_vec,u_vec,v_vec,0.75*plt_ps(4)/plt_ps(3),'color',clr_vec);
            end
            hold on;
        end

    elseif (opt_plt == 2) %----- Option 2: Surfaces -----%
        U_labels = ["\rho","{\it u}","{\it v}","{\it p}","\phi"]; 
        for k = 1:n_var+1
            nexttile(fig,k);
            hold on; cla;
            if (k == 5)
                s = surf(X,Y,phi);
            else
                s = surf(X,Y,squeeze(U(k,:,:)));
            end
            clr = colorbar;
            clr.Label.String = U_labels(k);
            %s.EdgeColor = 'interp';%[0.5 0.5 0.5];
            %view(2);
        end
    end
end
end