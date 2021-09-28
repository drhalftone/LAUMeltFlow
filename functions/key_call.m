%- File: 'key_call.m'
%- Purpose: Defines a string 'key' which specifies parameters within
%--- parameter vector 'prm'
key = "prm = {n_dim,n_var,n,dx,x_min,x_max,t_f,cfl,flg_fld,EoS,c_EoS,slvr,flg_BCs,n_nds,ICs_hdr,n_disp,n_out,flg_intrp,flg_wrt,wrt_nm,flg_plt,opt_plt,n_r,e_r,flg_anmt,n_anmt,t_anmt,wrt_prfx,wrt_sfx,n_rstrt,rd_nm,rd_prfx,rd_sfx,plt_ps,plt_wn,flg_vec,n_vec,t_0};";

%       1      2      3    4   5      6
%prm = {n_dim  n_var  n    dx  x_min  x_max ...
%       7    8    9        10   11     12    13
%       t_f  cfl  flg_fld  EoS  c_EoS  slvr  flg_BCs ...
%       14     15       16      17       18
%       n_nds  ICs_hdr  n_disp  n_out    flg_intrp ...
%       19       20      21       22       23   24
%       flg_wrt  wrt_nm  flg_plt  opt_plt  n_r  e_r ...
%       25        26      27      28        29
%       flg_anmt  n_anmt  t_anmt  wrt_prfx  wrt_sfx ...
%       30        31     32       33      34      35
%       flg_rstrt rd_nm  rd_prfx  rd_sfx  plt_ps  plt_wn
%       36      37    38
%       flg_vec n_vec t_0};
%end
