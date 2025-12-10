clear all

gam = 1.4;
dx = 0.01;
x = 0:dx:1;
n = length(x);
d = 0.5;
t_f = 0.01;
cfl = 0.95;

s = zeros(1,n);
for i = 1:n
    if (x(i) < d)
        U(:,i) = [10,10,1e5];
        s(i) = 0.5-x(i);
    else
        U(:,i) = [1,10,1e4];
        %s(i) = log(U(3,i)) - gam*log(U(1,i));
        s(i) = -0.5+x(i);
    end
    a(i) = sqrt(gam*U(3,i)/U(1,i));
end

t = 0; it = 0;
prm = {1,3,n,dx};
while (t < t_f)
    it = it + 1;
    dt = 1/2*cfl*dx/max(abs(U(2,:)) + a);
    t = t + dt;
    s = advc(prm,1,dt,U(2,:),s);
end

plot(x,s);
xlabel('$x$');
ylabel('$s$');
