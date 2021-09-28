


   %===== Calculate and Plot Speed of Sound (Optional) =====%  
if (flg_pltSoS)                 % Calculate speed of sound
    disp('Plotting speed of sound...');
    a = zeros(npt(1),npt(2));
    for i = 1:npt(1)
        for j = 1:npt(2)
            a(i,j) = sqrt(gam*U(4,i,j)/U(1,i,j));
        end
    end                         % Plot speed of sound
    if (flg_intrp)              % Interpolate speed of sound (Optional)
        a_out = interp2(X,Y,a,X_out,Y_out);
    else
        a_out = a;
    end
    plt_SoS(1,:,:) = a_out; plt_SoS(2,:,:) = U_out(2,:,:); 
    plt_SoS(3,:,:) = U_out(3,:,:);
    [~] = plot_2D(5,X_out,Y_out,plt_SoS);
end

if (opt_plt == 5)                       % Speed of sound plot option (5)
    U_labels = ["{\it a} [m/s]","{\it u} [m/s]", ...
    "{\it v} [m/s]","{\it p} [Pa]"];
    fig = figure('Position', [10 10 450 400]);
    quiver(X_vec,Y_vec,squeeze(u_vec(1,:,:)), ...
        squeeze(v_vec(1,:,:)),'color',[0.75 0.75 0.75])
    hold on
    contour(X(1,:,:),X(2,:,:),squeeze(U(1,:,:)))
    clr = colorbar;
    clr.Label.String = U_labels(1);
    xlabel('{\it x} [m]'), ylabel('{\it y} [m]')
    title('Speed of Sound');
end