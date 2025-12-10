clear all
%
n = 51;
xmin = -1;
xint = 0.5;
xmax = 1;
x = linspace(xmin,xmax,n);
dx = x(n) - x(n-1);
phimin = -1;
phimax = 1;
L = 0.1*(xmax-xmin);
%
for i = 1:n
    phi(i) = sin(pi/2*x(i));
    % if (x(i) < xint)
    %     phi(i) = phimin;
    % elseif (x(i) >= xint - L && x(i) <= xint + L)
    %     phi(i) = phimin + (phimax - phimin)/(xmax-xmin)*(x(i)-xint);
    % else
    %     phi(i) = phimax;
    % end
end
%
ndim = 1;
prm{1} = ndim;
prm{3} = n;
prm{4} = dx;
prm{24} = 1;
%
itmax = 400;
plot(x,phi);
%
for it = 1:itmax
    disp(it);
    figure(1)
    hold off
    plot(x,phi);
    xlabel('x')
    ylabel('\phi')
    hold off
    % pause
    phi = reinit_fast(prm,phi);
    pause(0.05);
end
