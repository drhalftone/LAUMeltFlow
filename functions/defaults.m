%- File: 'defaults.m'
%- Purpose: Assigns default simulation parameters if not specified in input
%--- file

if (exist('n_dim','var')), else, error('defaults.m: Number of spatial dimensions not specified'); end
if (exist('dx','var')), else, error('defaults.m: Grid step not specified'); end
if (exist('x_min','var')), else, error('defaults.m: Grid minimum boundarie not specified'); end
if (exist('x_max','var')), else, error('defaults.m: Grid maximum boundarie not specified'); end
if (exist('t_f','var')), else, t_f = 1; end
if (exist('cfl','var')), else, cfl = 0.9; end
if (exist('flg_fld','var')), else, error('defaults.m: Fluid types not specified'); end 
if (exist('EoS','var')), else, error('defaults.m: Equations of state not specified'); end
if (exist('c_EoS','var')), else, error('defaults.m: Equation of state parameters not specified'); end
if (exist('slvr','var')), else, error('defaults.m: Solver types not specified'); end
if (exist('flg_BCs','var')), else, flg_BCs = 0; end
if (length(flg_BCs) == 1)
    if (n_dim == 1)
        flg_BCs = flg_BCs*ones(1,2);
    elseif (n_dim == 2)
        flg_BCs = flg_BCs*ones(1,4);
    end
end
if (exist('n_nds','var')), else, n_nds = 0; end
if (exist('ICs_hdr','var')), else, ICs_hdr = "%------------------ (Simulation Description Unspecified) ----------------%"; end
if (exist('n_disp','var')), else, n_disp = 10; end
if (n_dim == 2)
    if (exist('n_out','var'))
        n_chk = size(n); 
        if (n_chk(1) == 1)
            n_out = [n_out,n_out]; 
        end
    else
        if (exist('dx_out','var'))
            if (length(dx_out) == 1)
                dx_out = [dx_out,dx_out]; 
            end
            n_out(1) = (x_max(1)-x_min(1))/dx_out(1)+1;
            n_out(2) = (x_max(2)-x_min(2))/dx_out(2)+1;
        end
    end
end
if (exist('n_out','var')), flg_intrp = 1; else, flg_intrp = 0; n_out = 0; end
if (exist('wrt_nm','var')), flg_wrt = 1; else, flg_wrt = 0; end
if (exist('opt_plt','var')), if (opt_plt > 0), flg_plt = 1; end, else, flg_plt = 0; end
if (exist('n_r','var')), else, n_r = 0; e_r = 0; end
if (exist('flg_anmt','var')), else, flg_anmt = 0; end
if (exist('n_anmt','var')), flg_anmt = 1; else, n_anmt = 5; end
if (exist('t_anmt','var')), flg_anmt = 1; else, t_anmt = 0.05; end
if (flg_plt), else, flg_anmt = 0; end
if (exist('wrt_prfx','var')), else, wrt_prfx = "data/"; end
if (exist('wrt_sfx','var')), else, wrt_sfx = ".d"; end
if (exist('n_rstrt','var')), else, n_rstrt = 0; end 
if (exist('rd_nm','var')), else, rd_nm = "flow_unnamed"; end
if (exist('rd_prfx','var')), else, rd_prfx = "data/"; end
if (exist('rd_sfx','var')), else, rd_sfx = ".flo"; end
if (exist('plt_ps','var'))
else
    if (n_dim == 1)
        plt_ps = [10 10 750 600];
    elseif(n_dim == 2)
        plt_ps = [10 10 1350 750];
    end
end
if (exist('plt_wn','var'))
else
    plt_wn = 2;
end
if (exist('n_vec','var')), flg_vec = 2; if (length(n_vec) == 1), n_vec = [n_vec,n_vec]; end, end
if (exist('flg_vec','var')), else, flg_vec = 0; n_vec = 0; end
if (exist('t_0','var')), else, t_0 = 0; end
