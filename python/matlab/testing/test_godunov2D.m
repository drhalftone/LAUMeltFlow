clear all

gam = 1.4;
t_f = 0.008;
cfl = 0.95;

n_dim = 2;                              % -> # of spatial dimensions
dx = [0.02,0.02];                       % -> x and y-grid spacings [m]
x(1,:) = 0:dx(1):1;                     % -> x-grid boundaries [m]
x(2,:) = 0:dx(2):1;                     % -> y-grid boundaries [m]
d = [0.5,0.5,0.5];                      % -> Dimensions [m]
U_r(1,:) = [1.226,-100,-100,1.0e5];           % -> Region (1) variables
U_r(2,:) = [10,-100,-100,1.01e5];         % -> Region (2) variables

                   %===== Build Initial Conditions =====% 
n(1) = length(x(1,:)); n(2) = length(x(2,:));
for i = 1:n(1)                        % Assign properties
    for j = 1:n(2)
        if (norm([x(1,i)-d(2),x(2,j)-d(3)]) <= d(1)/2)
            U(:,i,j) = U_r(2,:);
            phi(i,j) = -1;
        else
            U(:,i,j) = U_r(1,:);
            phi(i,j) = 1;
        end
        %phi(i,j) = sqrt((x(1,i)-d(2))^2 + (x(2,j)-d(3))^2) - 1/2*d(1); %
        a(i,j) = sqrt(gam*U(4,i,j)/U(1,i,j));
        s(i,j) = log(U(4,i,j)) - gam*log(U(1,i,j));
    end
end

%phi = zeros(n(1),n(2));             % Build level set
[X(1,:,:),X(2,:,:)] = meshgrid(x(1,:),x(2,:));

t = 0; it = 0;
prm = {n_dim,n_dim+2,n,dx};
% while (t < t_f)
%     it = it + 1;
%     [dphidx,dphidy] = gradient(phi,dx(1),dx(2));
%     for i = 1:n(1)
%         for j = 1:n(2)
%             N(:,i,j) = [dphidx(i,j),dphidy(i,j)]/norm([dphidx(i,j),dphidy(i,j)]);
%             if (dphidx(i,j) == 0 && dphidy(i,j) == 0)
%                 N(1,i,j) = 0; N(2,i,j) = 0;
%             end
%             V(i,j) = norm([U(2,i,j),U(3,i,j)]);
%         end
%     end
%     dt = 1/2*cfl*min(dx)/max(abs(V + a),[],'all');
%     t = t + dt;
%     s = advc(prm,1,dt,N,s);
%     %s = advc(prm,1,dt,U(2:3,:,:),s);
% end

%nx = squeeze(N(1,:,:)); ny = squeeze(N(2,:,:));
Y = squeeze(X(2,:,:)); X = squeeze(X(1,:,:));
%Xq = zeros(n(1),n(2)); Yq = zeros(n(1),n(2));
k = 0;
for i = 1:n(1)
    for j = 1:n(2)
        if (phi(i,j) <= 0)
            k = k + 1;
            xq(k) = X(i,j); yq(k) = Y(i,j); ss(k) = s(i,j);
        elseif (phi(i,j) > 0)
            %Xq(i,j) = []; Yq(i,j) = [];
        end
    end
end
F = scatteredInterpolant(xq(:),yq(:),ss(:));
sss = F(X,Y);
%quiver(X,Y,nx,ny);
surf(X,Y,sss);
%surf(X,Y,s);
xlabel('$x$');
ylabel('$y$');
zlabel('$s$');
zlim([0 10]);