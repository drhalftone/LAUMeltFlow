%- File: grid_setup.m
%- Purpose: Determine # of primitive/conserved variables 'n_var' and
%--- allocate grid before assigning initial conditions to 'U', 'phi

n_var = n_dim + 2;

if (n_dim == 1)
    if (exist('n','var'))
        x = linspace(x_min,x_max,n);
        dx = x(2) - x(1);
    else
        if (exist('dx','var'))
            x = x_min:dx:x_max;
            n = length(x);
        end
    end
    U = zeros(n_var,n);
    phi = zeros(1,n);
elseif (n_dim == 2)
    if (exist('n','var'))
        n_chk = size(n); 
        if (n_chk(1) == 1)
            n = [n,n]; 
        end
    else
        if (exist('dx','var'))
            if (length(dx) == 1)
                dx = [dx,dx]; 
            end
        end
    end
    if (exist('n','var'))
        xx = linspace(x_min(1),x_max(1),n);
        yy = linspace(x_min(2),x_max(2),n);
        dx(1) = xx(2) - xx(1); dx(2) = yy(2) - yy(1);
    else
        xx = x_min(1):dx(1):x_max(1);
        yy = x_min(2):dx(2):x_max(2);
        n(1) = length(xx); n(2) = length(yy);
    end
    x = zeros(n_dim,max(length(xx),length(yy)));
    x(1,1:n(1)) = xx; x(2,1:n(2)) = yy;
    U = zeros(n_var,n(1),n(2));
    phi = zeros(n(1),n(2));
end