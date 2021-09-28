function print_term(prm,flg)
%- Purpose: Prints relevant data to terminal

                        %===== Variables =====%
global t t_wall it it_r
[n_dim,~,n,dx,x_min,x_max,t_f,cfl,ICs_hdr,n_r,e_r] = deal(prm{1:8},prm{15},prm{23:24});

                         %===== Intrinsics =====%
dim_max = 2;                            % Maximum # of dimensions in code
ln_max = 9;                             % Maximum # of lines to store

                           %===== Flags =====%
flg_skp_prv = [1,0,0,0,0,0,0,0,1];      % Skip previous line before printing?
flg_skp_nxt = [0,1,0,0,0,0,0,1,1];      % Skip next line after printing?
flg_hdr = [1,1,1,1,1,1,0,0,1];          % Print header?
flg_ln = [0,0,1,1,1,0,1,0,0];           % Print line of data?
                           
                          %===== Headers =====%
hdr(1,1) = "%======================= Ghost-fluid Method Solver ======================%";
if (n_dim == 1), hdr(1,2) = ICs_hdr; end
hdr(1,3) = "%----- Final Time ---------------- CFL --------------- Grid Size --------%";
hdr(1,4) = "%------------------ dx -------------------- Grid Points -----------------%";
hdr(1,5) = "%------------- # Iter/Reinit ------------ Reinit Tolerance --------------%";
hdr(1,6) = "%--- Iteration --- # Reinit ----- Simulation Time ----- Wall Time [s] ---%";
hdr(1,9) = "%============================= End of Output ============================%";

hdr(2,1) = "%======================= Ghost-fluid Method Solver ======================%";
if (n_dim == 2), hdr(2,2) = ICs_hdr; end
hdr(2,3) = "%------ Final Time ---------------- CFL ---------------- Grid Size ------%";
hdr(2,4) = "%---------- dx -------------------- dy ----------------- Grid Points ----%";
hdr(2,5) = "%------------- # Iter/Reinit ------------ Reinit Tolerance --------------%";
hdr(2,6) = "%--- Iteration --- # Reinit ----- Simulation Time ----- Wall Time [s] ---%";
hdr(2,9) = "%============================= End of Output ============================%";

                           %===== Lines =====%
switch n_dim
    case 1
ln{1,3}= sprintf('%16f %24f %22f \n',[t_f,cfl,x_max-x_min]);
ln{1,4} = sprintf('%26f %25d \n',[dx,n]);
ln{1,5} = sprintf('%24d %31e \n',[n_r,e_r]);
ln{1,7} = sprintf('%12d %13d %19f %20f',[it,it_r,t,t_wall]);
    case 2
ln{2,3} = sprintf('%17f %23f %18f x %f \n',[t_f,cfl,x_max(1)-x_min(1),x_max(2)-x_min(2)]);
ln{2,4} = sprintf('%17f %23f %19d x %d \n',[dx(1),dx(2),n(1),n(2)]);
ln{2,5} = sprintf('%24d %31e \n',[n_r,e_r]);
ln{2,7} = sprintf('%12d %13d %19f %20f',[it,it_r,t,t_wall]);
end

for dim = 1:dim_max
   for j = 1:ln_max
       if (dim == n_dim && j == flg)
          if (flg_skp_prv(j)), fprintf('\n'); end
          if (flg_hdr(j)), disp(hdr(dim,j)); end
          if (flg_ln(j)), disp(ln{dim,j}); end
          if (flg_skp_nxt(j)), fprintf('\n'); end
      end
   end
end