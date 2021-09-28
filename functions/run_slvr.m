function WW = run_slvr(prm,dt,X,phi,UU,WW)
%- Purpose: Pass primitive and conserved variables to solver for time step
%--- updates

[n_dim,slvr] = deal(prm{1},prm{12});

for k = 1:2
    if (n_dim == 1)
        U = squeeze(UU(k,:,:)); W = squeeze(WW(k,:,:));
        WW(k,:,:) = feval(slvr(k),prm,k,dt,X,phi,U,W);        
    elseif (n_dim == 2)
        U = squeeze(UU(k,:,:,:)); W = squeeze(WW(k,:,:,:));
        WW(k,:,:,:) = feval(slvr(k),prm,k,dt,X,phi,U,W);       
    end

end