function [prm,X,U,phi] = mesh_ICs(fl_in,key)
%- Purpose: Generate grid and call initial conditions for GFM solver

eval(fl_in);                            % Load parameters, ICs
defaults;                               % Evaulate default parameters if not specified
eval(key);                              % Load parameters into 'prm' cell vector

                        %===== Mesh =====%   
if (n_dim == 1)                         % 1D mesh
    X = x;  
elseif (n_dim == 2)                     % 2D mesh
    [XX,YY] = meshgrid(x(1,1:n(1)),x(2,1:n(2)));
    X(1,:,:) = XX'; X(2,:,:) = YY';
end

