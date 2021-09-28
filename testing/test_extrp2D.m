clear all

gam = 1.4;
t_f = 0.008;
cfl = 0.95;

n_dim = 2;                              % -> # of spatial dimensions
dx = [0.02,0.02];                       % -> x and y-grid spacings [m]
x(1,:) = 0:dx(1):1;                     % -> x-grid boundaries [m]
x(2,:) = 0:dx(2):1;                     % -> y-grid boundaries [m]

                   %===== Build Initial Conditions =====% 
n(1) = length(x(1,:)); n(2) = length(x(2,:));

for i = 1:n(1)                        % Assign properties
    for j = 1:n(2)
        f(i,j) = 0;
        phi(i,j) = 1;
    end
end

d = [0.1,0.2,0.2];                      % -> Dimensions [m]
for i = 1:n(1)                        % Assign properties
    for j = 1:n(2)
        if (norm([x(1,i)-d(2),x(2,j)-d(3)]) <= d(1)/2)
            f(i,j) = 2;
            phi(i,j) = -1;
        end
    end
end

d = [0.1,0.8,0.8];                      % -> Dimensions [m]
for i = 1:n(1)                        % Assign properties
    for j = 1:n(2)
        if (norm([x(1,i)-d(2),x(2,j)-d(3)]) <= d(1)/2)
            f(i,j) = 8;
            phi(i,j) = -1;
        end
    end
end

[X(1,:,:),X(2,:,:)] = meshgrid(x(1,:),x(2,:));

t = 0; it = 0;
prm = {n_dim,n_dim+2,n,dx};

fex = extrp(prm,X,phi,f);

Y = squeeze(X(2,:,:)); X = squeeze(X(1,:,:));
figure
%surf(X,Y,f); hold on;
surf(X,Y,fex);
%legend(["original","extrapolated"]);
xlabel('$x$');
ylabel('$y$');
zlabel('$f$');
%zlim([0 10]);