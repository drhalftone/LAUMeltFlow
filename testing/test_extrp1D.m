clear all

gam = 1.4;
dx = 0.01;
x = 0:dx:1;
n = length(x);
t_f = 0.01;
cfl = 0.95;

s = zeros(1,n);
for i = 1:n
    if (x(i) < 0.2)
        f(i) = 0.2;
        phi(i) = -1;
    elseif (x(i) > 0.8)
        f(i) = 0.8;
        phi(i) = -1;
    else
        f(i) = 0;
        phi(i) = 1;
    end
end

t = 0; it = 0;
prm = {1,3,n,dx};
f = extrp(prm,x,phi,f);

plot(x,f);
xlabel('$x$');
ylabel('$f$');
