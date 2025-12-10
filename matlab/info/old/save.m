
% if (flg_smth)                           % Smooth out level set function
%     dphidx = dfdx_FDunfrm(n_dim,2,dx,phi);
%     for i = 1:npt
%         if (i > 1 && i < npt)       % Find liquid regions
%             if (phi(i+1) <= 0 && phi(i) > 0)
%                 i_l = i+1;          % 
%             end
%             if (phi(i-1) <= 0 && phi(i) > 0)
%                 i_r = i-1;          % 
%                 if (i_l == i_r)     % One liquid point case
%                     
%                 else
%                     x_mid = 1/2*(x(i_l) + x(i_r));
%                     a_phi = 1/(2*(x(i_r) - x_mid)*dphidx(i_r));
%                     h_phi = a_phi*(x(i_r) - x_mid)^2 - phi(i_r);
%                     for j = i_l:i_r % 
%                         phi(i) = a_phi*(x(i)-x_mid)^2 - h_phi;
%                     end
%                 end
%             end
%         end
%     end
% end

%flg_pltSoS = 0;                         % -> Plot speed of sound?