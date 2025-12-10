function wrt_data(prm,X_out,U_out,phi_out,wrt_fl)
%- Purpose: Writes out 1D/2D CFD primitive variable data to a file
%- Variables:
%--- U = [rho u (v) p]^T Array of primitive variables
%- Assumptions:
%--- Cartesian grid uniform in each direction
%- Format: 
%--- first line = header
%--- data = [x y rho u (v) p] (Write by line)
%--- order = [x(1) y(1) <-- U(1,1)^T --> ]
%---         [x(2) y(1) <-- U(2,1)^T --> ]
%---         [x(3) y(1) <-- U(3,1)^T --> ]
%---         [x(1) y(2) <-- U(1,2)^T --> ]
%---         [x(2) y(2) <-- U(2,1)^T --> ] ...

[n_dim,n_var,n,n_out,flg_intrp] = deal(prm{1:3},prm{17:18});

if (flg_intrp == 0)
    n_out = n;
end

fprintf("Writing flow field (file = '%s')... \n",wrt_fl);
[fid,msg] = fopen(wrt_fl,'wt');         % Open file
    
switch n_dim
    case 1             %===== 1D Case =====% 
        data ...                    % Allocate output matrix
            = zeros(n_out,n_var+n_dim+1);
            
        for i = 1:n_out
           data(i,:) = [X_out(1,i),U_out(1,i) ...
               ,U_out(2,i),U_out(3,i),phi_out(i)];
        end
        fprintf(fid,'x rho u p phi \n');% Write header to file
        fprintf(fid ...                 % Write data to file
            ,'%f %f %f %f %f \n',data');
    case 2             %===== 2D Case =====% 
        data ...                    % Allocate output matrix
            = zeros(n_out(1)*n_out(2),n_var+n_dim+1);
        k = 0; 
        for i = 1:n_out(1)
           for j = 1:n_out(2)
               k = k + 1;               % Expand data for writing
               data(k,:) = [X_out(1,i,j),X_out(2,i,j),U_out(1,i,j) ...
                   ,U_out(2,i,j),U_out(3,i,j),U_out(4,i,j),phi_out(i,j)];
           end
        end
        fprintf(fid,'x y rho u v p phi \n');% Write header to file
        fprintf(fid ...                 % Write data to file
            ,'%f %f %f %f %f %f %f \n',data');
end
fclose(fid);                            % Close file