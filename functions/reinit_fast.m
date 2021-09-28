function [phi] = reinit_fast(prm,phi0)
%- Purpose: Reinitialize a signed-distance level set function phi
%-Variables:
%--- e_r = Convergence tolerance

    [n_dim,~,n,dx,e_r] = deal(prm{1:4},prm{24});
    if (n_dim == 1)
    else
        error("2D and 3D not currently supported");
    end
    phi = phi0; it = 0; e = pi*e_r;        % Initialize solver
    itmax = 1;
    xlo = -1; xhi = 1;
    dtloc = 1/2*min(dx);
    zero_tol = 0;
    numGhost = 4;
    ilo = numGhost+1;
    ihi = n-numGhost;
    numGhostTot = 2*numGhost;
    nReal = n-numGhostTot;
    x = linspace(xlo,xhi,n);
    xReal = linspace(xlo+numGhost*dx,xhi-numGhost*dx,nReal);
    while (it < itmax)
        it = it + 1;
        endslope(1:n) = gradient(phi,dx);
        gradphi(ilo:ihi) = gradient(phi(ilo:ihi),dx);
        for i = 1:numGhost
            phicur(i) = phi(ilo)-(ilo-i)*dx*endslope(ilo);
            phicur(i+ihi) = phi(ihi)+(i)*dx*endslope(ihi);
        end
        RHS(1,ilo:ihi) = zeros(1,nReal);
        phicur(ilo:ihi) = phi(ilo:ihi);
        %
        % Compute derivative at cell edges
        phi_minus(ilo:ihi) = derWENOMinus(phi,dx,nReal,numGhost);
        phi_plus(ilo:ihi) = derWENOPlus(phi,dx,nReal,numGhost);
        %
        % Compute reinitialization equation RHS
        for i = 1+numGhost:n-numGhost
            phi_i = phicur(i);
            %
            % Godunov selection process
            grad_phi_plus = phi_plus(i);
            grad_phi_minus = phi_minus(i);
            % if (phi_i > 0)
            %     grad_phi_plus = max(-grad_phi_plus,0);
            %     grad_phi_minus = max(grad_phi_minus,0);
            % else
            %     grad_phi_minus = max(-grad_phi_minus,0);
            % end
            % grad_phi_star = max(grad_phi_plus,grad_phi_minus);
            grad_phi_star = 1/2*(grad_phi_plus+grad_phi_minus);
            grad_vec(i) = grad_phi_star;
            %
            % Compute RHS using smoothed sign of level set
            if (abs(phi_i) >= zero_tol)
                norm_grad_phi = abs(grad_phi_star);
                sgn_phi = phi_i/sqrt(phi_i^2 + norm_grad_phi^2 + dx^2);
                RHS(i) = sgn_phi*(1 - norm_grad_phi);
                % RHS(i) = sgn_phi*(1 - norm_grad_phi(i));
            else
                RHS(i) = 0;
            end
        end
        %
        % (Optional) Plot derivative
        % figure(2);
        % plot(xReal,gradphi(ilo:ihi));
        % hold on;
        % plot(xReal,grad_vec(ilo:ihi));
        % xlabel('x');
        % ylabel('d\phi/dx');
        % legend('exact','computed');
        % hold off;
        % RHS(1:numGhost) = 0;
        % RHS(n-numGhost+1:n) = 0;
        %
        % Advance in time
        phi = phicur + dtloc*RHS;
    end
    %
    function der_minus = derWENOMinus(f,dx,n,numGhost)
    %
        n_tot = n + 2*numGhost;
        %
        % Undivided differences
        for i = 1:n_tot-1
            D1(i) = (f(i+1) - f(i))/dx;
        end
        %
        for i = 1:n
            k = i;
            for j = 1:5
                v(j) = D1(k+j);
                 dx_array(j) = v(j)^2;
            end
            epsln = 1e-6*max(dx_array) + 1e-99;
            %
            S(1) = 13/12*(v(1)-2*v(2)+v(3))^2 + 1/4*(v(1)-4*v(2)+3*v(3))^2;
            S(2) = 13/12*(v(2)-2*v(3)+v(4))^2 + 1/4*(v(2)-v(4))^2;
            S(3) = 13/12*(v(3)-2*v(4)+v(5))^2 + 1/4*(3*v(3)-4*v(4)+v(5))^2;
            %
            alpha(1) = (1/10)/(S(1)+epsln)^2;
            alpha(2) = (6/10)/(S(2)+epsln)^2;
            alpha(3) = (3/10)/(S(3)+epsln)^2;
            alpha_tot = sum(alpha);
            %
            f_x(1) = v(1)/3 - 7*v(2)/6 + 11*v(3)/6;
            f_x(2) = -v(2)/6 + 5*v(3)/6 + v(4)/3;
            f_x(3) = v(3)/3 + 5*v(4)/6 - v(5)/6;
            %
            % Final derivative
            der_minus(i+numGhost) = (alpha(1)*f_x(1) + alpha(2)*f_x(2) + alpha(3)*f_x(3))/alpha_tot;
            %
        end
        der_minus(1:numGhost) = [];
        %
    end
    %
    function der_plus = derWENOPlus(f,dx,n,numGhost)
    %
        n_tot = n + 2*numGhost;
        %
        % Undivided differences
        for i = 1:n_tot-1
            D1(i) = (f(i+1) - f(i))/dx;
        end
        %
        for i = 1:n
            k = i+1;
            for j = 1:5
                v(j) = D1(k+6-j);
                dx_array(j) = v(j)^2;
            end
            epsln = 1e-6*max(dx_array) + 1e-99;
            %
            S(1) = 13/12*(v(1)-2*v(2)+v(3))^2 + 1/4*(v(1)-4*v(2)+3*v(3))^2;
            S(2) = 13/12*(v(2)-2*v(3)+v(4))^2 + 1/4*(v(2)-v(4))^2;
            S(3) = 13/12*(v(3)-2*v(4)+v(5))^2 + 1/4*(3*v(3)-4*v(4)+v(5))^2;
            %
            alpha(1) = (1/10)/(S(1)+epsln)^2;
            alpha(2) = (6/10)/(S(2)+epsln)^2;
            alpha(3) = (3/10)/(S(3)+epsln)^2;
            alpha_tot = sum(alpha);
            %
            f_x(1) = v(1)/3 - 7*v(2)/6 + 11*v(3)/6;
            f_x(2) = -v(2)/6 + 5*v(3)/6 + v(4)/3;
            f_x(3) = v(3)/3 + 5*v(4)/6 - v(5)/6;
            %
            % Final derivative
            der_plus(i+numGhost) = (alpha(1)*f_x(1) + alpha(2)*f_x(2) + alpha(3)*f_x(3))/alpha_tot;
            %
        end
        der_plus(1:numGhost) = [];
        %
    end
    %
end
