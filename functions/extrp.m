function I = extrp(fld,prm,X,phi,I)
%- Purpose: Extrapolates (or interpolates) a data set 'I' within the domain
%--- X{phi<=0} to remainder of grid where phi>0
%- Variables:
%--- fld = Which fluid domain to extrapolate from

[n_dim,n] = deal(prm{1},prm{3});

if (n_dim == 1)         %===== 1D Case =====%
    k = 0;
    for i = 1:n
        if (fld == 1 && phi(i) > 0) || (fld == 2 && phi(i) <= 0)
            k = k + 1;
            x_in(k) = X(i); I_in(k) = I(i);
        end
    end
    if (k > 0)
        F = griddedInterpolant(x_in(:),I_in(:));
        I = F(X);
    end
    
elseif (n_dim == 2)     %===== 2D Case =====%
    k = 0;
    for i = 1:n(1)
        for j = 1:n(2)
            if ((fld == 1 && phi(i,j) > 0) || (fld == 2 && phi(i,j) <= 0))
                k = k + 1;
                x_in(k) = X(1,i,j); y_in(k) = X(2,i,j); I_in(k) = I(i,j);
            end
        end
    end
    if (k > 0)
        F = scatteredInterpolant(x_in(:),y_in(:),I_in(:));
        I = F(squeeze(X(1,:,:)),squeeze(X(2,:,:)));
    end
end

i;