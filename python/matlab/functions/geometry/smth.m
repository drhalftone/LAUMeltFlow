function f = smth(k,n_dim,dx,x,f)
%- Purpose: Smooth out a function on grid x = [x,(y)] controlled by an
%--- integer smoothing parameter k > 0
dx_q = dx.*k;
switch n_dim
    case 1
        x_q = min(x):dx_q:max(x);
        f_q = interp1(x,f,x_q,'spline');
        f = interp1(x_q,f_q,x,'spline');
    case 2
        [X,Y] = meshgrid(x(1,:),x(2,:));
        x_q = min(x(1,:)):dx_q(1):max(x(1,:));
        y_q = min(x(2,:)):dx_q(2):max(x(2,:));
        [X_q,Y_q] = meshgrid(x_q,y_q);
        f_q = interp2(X,Y,f,X_q,Y_q,'spline');
        f = interp2(X_q,Y_q,f_q,X,Y,'spline');
end