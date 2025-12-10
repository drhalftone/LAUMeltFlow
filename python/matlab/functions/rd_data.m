%====================== CFD External Data Reader  =======================%

%- Purpose: Reads in 1D/2D CFD primitive variable data for plotting or other
%-- operations
%- Variables:
%--- U = [rho u (v) p]^T Array of primitive variables
%- Assumptions:
%--- Cartesian grid uniform in each direction
%- Format: 
%--- first line = header
%--- data = [x y rho u (v) p] (Reads by line)
%--- order = [x(1) y(1) <-- U(1,1)^T --> ]
%---         [x(2) y(1) <-- U(2,1)^T --> ]
%---         [x(3) y(1) <-- U(3,1)^T --> ]
%---         [x(1) y(2) <-- U(1,2)^T --> ]
%---         [x(2) y(2) <-- U(2,1)^T --> ] ...

                       %===== Intrinsics =====%
close all                               % Clear figures
flg_rd = 1;                             % -> Read data from file?
if (flg_rd), clear all; flg_rd = 1; end % Clear stored variables
dum = 0;                                % Assign dummy variable
n_dim = 2;                              % # of dimensions
n_prp = 4;                              % # of primitive/conserved variables

                        %===== Options =====%
flg_plt = 1;                            % -> Plot results?
opt_plt = 3;                            % -> Plotting option (1=velocity 
                                            % vector,2=contour,3=contour with 
                                            % velocity vectors,4=surface)
flg_pltSoS = 0;                         % -> Plot speed of sound? 
                                            % (Specific heat ratio required)
gam = 1.4;                              % Specific heat ratio for SoS calculation
rd_prfx = 'data/';                      % -> File prefix for data reading
rd_nm = 'flow';                         % -> File name for data reading
rd_sfx = '.d';                          % -> File suffix for data reading

                        %===== Read Data =====%  
if (flg_rd)
    rd_fl = [rd_prfx rd_nm rd_sfx];     % Read file name
    [fid,msg] = fopen(rd_fl,'r');       % Read the file
    fgetl(fid);                         % Skip header line
    rd_sz = [n_dim+n_prp,Inf];          % Data matrix size
    rd_data = fscanf(fid,'%f %f %f %f %f %f',rd_sz);
    fclose(fid);
    rd_data = rd_data';
    x_in = rd_data(:,1);                % Reconstruct grid points
    y_in = rd_data(:,2);
    npt_in = length(x_in);
    dx = x_in(2) - x_in(1);             
    x_min = min(x_in); x_max = max(x_in);
    x = x_min:dx:x_max;                 % Reconstruct x points
    dy = y_in(length(x)+1) - y_in(length(x));
    y_min = min(y_in); y_max = max(y_in);
    y = y_min:dy:y_max;                 % Reconstruct y points
    [X,Y] = meshgrid(x,y);              % Generate grid
    npt = [length(x),length(y)];        % # of grid points = [npt_x,npt_y]^T
    l = 0; U = zeros(n_prp,npt(1),npt(2));
    for i = 1:npt(1)                    % Reconstruct primitive variables
        for j = 1:npt(2)
            l = l + 1;
            for k = 1:n_prp
                U(k,i,j) = rd_data(l,k+n_dim);
            end
        end
    end
end

                      %===== Plot Results =====%  
if (flg_plt), [dum] = plot_2D(opt_plt,X,Y,U); end
if (flg_plt && flg_pltSoS)              % Calculate speed of sound (Optional)
    a = zeros(npt(1),npt(2));
    for i = 1:npt(1)
        for j = 1:npt(2)
            a(i,j) = sqrt(gam*U(4,i,j)/U(1,i,j));
        end
    end
    plt_SoS(1,:,:) = a; 
    plt_SoS(2,:,:) = U(2,:,:); 
    plt_SoS(3,:,:) = U(3,:,:);          % Plot speed of sound
    [dum] = plot_2D(5,X,Y,plt_SoS);
end