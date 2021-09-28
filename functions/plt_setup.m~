function fig = plt_setup(prm)
%- Purpose: Set up plot for primitive variable data 'U' over grid '(X,Y)'
%--- and level set 'phi'
%- Method: Plotting options set by 'opt_plt'
%- Variables:
%--- opt_plt = 1D flag, 2D Plotting option (1=contour with velocity vectors,2=surface)
%--- U = [rho u (v) p]^T Array of primitive variables

[n_dim,n_var,opt_plt,plt_ps,plt_wn] = deal(prm{1:2},prm{22},prm{34:35});

if (n_dim == 1)           %===== 1D Case =====% 
    U_labels = ["$\rho$","$u$","$p$","$\phi$"];
    figure('Position',plt_ps);
    plt_hn = ceil((n_var+1)/plt_wn);
    fig = tiledlayout(plt_hn,plt_wn);
    txt_x = xlabel(fig,"$x$"); txt_x.Interpreter = 'latex'; 
    for k = 1:n_var+1
        nexttile;
        txt_y = ylabel(U_labels(k)); txt_y.Interpreter = 'latex';
    end 
    
elseif (n_dim == 2)       %===== 1D Case =====% 
    if (opt_plt == 1)                   % Contour with velocity vectors
        plt_hn = n_var - plt_wn;
        figure('Position',plt_ps);
        fig = tiledlayout(plt_hn,plt_wn);
        txt_x = xlabel(fig,"$x$"); txt_x.Interpreter = 'latex'; 
        txt_y = ylabel(fig,"$y$"); txt_y.Interpreter = 'latex'; 
        for k = 1:3
            nexttile;
            ylabel(" ");
        end
    elseif (opt_plt == 2)
        plt_hn = floor((n_var+2)/plt_wn);
        figure('Position',plt_ps);
        fig = tiledlayout(plt_hn,plt_wn);
        txt_x = xlabel(fig,"$x$"); txt_x.Interpreter = 'latex'; 
        txt_y = ylabel(fig,"$y$"); txt_y.Interpreter = 'latex'; 
        for k = 1:n_var+1
            nexttile;
            view(2);
            %view(90,0);
            grid on;
            txt_z = zlabel(" "); txt_z.Interpreter = 'latex'; 
        end
    end
end
txt_t = title(fig,'Flow Field'); txt_t.Interpreter = 'latex';