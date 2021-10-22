function a = state_var(prm,var,sz,phi,H)
%- Purpose: Computes some state variable or array of variables 'a' named as
%--- 'var' over entire grid X=(X,(Y)). Uses array of primitive or conserved
%--- variables 'H' 

[n_dim,~,n,EoS,c_EoS,n_nds] = deal(prm{1:3},prm{10:11},prm{14});
c_EoS = cell2mat(c_EoS);

%f_call ...                              % Construct function calls
%    = [append(var,'_',EoS(1)),append(var,'_',EoS(2))];     
f_1 = str2func(append(var,'_',EoS(1)));
f_2 = str2func(append(var,'_',EoS(1)));

switch n_dim
    case 1             %===== 1D Case =====% 
        a = zeros(sz,n);                % Allocate array on grid
        for i = 1:n%,n_nds)
            if (phi(i) > 0 && EoS(1) ~= "none")
                %a(:,i) = feval(f_call(1),n_dim,c_EoS(1),H(:,i));
                a(:,i) = f_1(n_dim,c_EoS(1),H(:,i));
            elseif(phi(i) <= 0 && EoS(2) ~= "none")
                %a(:,i) = feval(f_call(2),n_dim,c_EoS(2),H(:,i));
                a(:,i) = f_2(n_dim,c_EoS(1),H(:,i));
            end
        end
        
    case 2             %===== 2D Case =====% 
        a = zeros(sz,n(1),n(2));        % Allocate array on grid
        for i = 1:n(1)%,n_nds)
            for j = 1:n(2)
                if (phi(i,j) > 0 && EoS(1) ~= "none")
                    %a(:,i,j) = feval(f_call(1),n_dim,c_EoS(1),H(:,i,j));
                    a(:,i,j) = f_1(n_dim,c_EoS(1),H(:,i,j));
                elseif(phi(i,j) <= 0 && EoS(2) ~= "none")
                    %a(:,i,j) = feval(f_call(2),n_dim,c_EoS(2),H(:,i,j));
                    a(:,i,j) = f_2(n_dim,c_EoS(1),H(:,i,j));
                end                
            end
        end
        a = squeeze(a);
end
