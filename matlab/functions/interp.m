function [X_out,U_out,phi_out] = interp(prm,X,U,phi)
%- Purpose: Interpolate grid primitive variable data 'U' on '(X,Y)'
%--- to a different data set 'U_out' on grid '(X_out,Y_out)'
%- Method: Constructs rectangular uniform grid of size [npt_x,npt_y] = n_out
%- Variables:
%--- U = [rho u (v) p]^T Array of primitive variables

[n_dim,n_var,x_min,x_max,n_out] = deal(prm{1:2},prm{5:6},prm{17});

if (n_dim == 1)          %===== 1D Case =====% 
    U_out = zeros(n_var,n_out);   % Allocate output variables
    x_out = linspace(x_min,x_max,n_out); % Output grid
    X_out = x_out;
    for k = 1:n_var                 % Interpolate
       U_out(k,:) = interp1(X,squeeze(U(k,:)),X_out);
    end
    phi_out = interp1(X,phi,X_out);

elseif (n_dim == 2)       %===== 2D Case =====% 
    Y = squeeze(X(2,:,:)); X = squeeze(X(1,:,:));
    U_out = zeros(n_var,n_out(1),n_out(2)); % Allocate output variables
    for dim = 1:n_dim
        x_out(dim,1:n_out(dim)) = linspace(x_min(dim),x_max(dim),n_out(dim));
    end
    [X_p,Y_p] = meshgrid(x_out(1,1:n_out(1)),x_out(2,1:n_out(2))); % Output mesh
    %X_p = X_p; Y_p = Y_p;
    for k = 1:n_var                     % Interpolate
       U_out(k,:,:) = interp2(X',Y',squeeze(U(k,:,:))',X_p,Y_p)';
    end
    phi_out = interp2(X',Y',phi',X_p,Y_p)';
    X_out(1,:,:) = X_p'; X_out(2,:,:) = Y_p';
end