%% 2D Electrostatic Potential from Point Charges
% Computes and visualizes the electrostatic potential field from
% randomly placed point charges using the analytical 2D Coulomb formula.
%
% Physics: phi(r) = -sum_i q_i * ln(|r - r_i|)
% This satisfies Poisson's equation: nabla^2 phi = -rho/epsilon_0

clear; clc; close all;

%% Parameters
nx = 128;               % Grid resolution
ny = 128;
domain = [0, 1, 0, 1];  % [xmin, xmax, ymin, ymax]
n_particles = 20;       % Number of point charges
epsilon = 2/nx;         % Regularization distance (avoid singularity)

%% Generate random particles
rng(42);  % For reproducibility

% Particle positions (avoid edges)
margin = 0.1;
particles.x = margin + (1-2*margin) * rand(n_particles, 1);
particles.y = margin + (1-2*margin) * rand(n_particles, 1);

% Random charges in [-1, 1]
particles.q = 2*rand(n_particles, 1) - 1;

fprintf('Generated %d particles\n', n_particles);
fprintf('  Total charge: %.3f\n', sum(particles.q));
fprintf('  Positive charges: %d\n', sum(particles.q > 0));
fprintf('  Negative charges: %d\n', sum(particles.q < 0));

%% Create computational grid
dx = (domain(2) - domain(1)) / nx;
dy = (domain(4) - domain(3)) / ny;

x = linspace(domain(1) + dx/2, domain(2) - dx/2, nx);
y = linspace(domain(3) + dy/2, domain(4) - dy/2, ny);
[X, Y] = meshgrid(x, y);

%% Compute charge density (scatter particles to grid)
rho = zeros(ny, nx);

for p = 1:n_particles
    % Find nearest grid cell
    ix = round((particles.x(p) - domain(1)) / dx);
    iy = round((particles.y(p) - domain(3)) / dy);

    % Clamp to valid range
    ix = max(1, min(nx, ix));
    iy = max(1, min(ny, iy));

    % Deposit charge (nearest-cell assignment)
    rho(iy, ix) = rho(iy, ix) + particles.q(p) / (dx * dy);
end

%% Compute analytical potential
% phi(r) = -sum_i q_i * ln(|r - r_i|)
% With regularization: |r - r_i| -> max(|r - r_i|, epsilon)

phi = zeros(ny, nx);

for p = 1:n_particles
    % Distance from this particle to all grid points
    r = sqrt((X - particles.x(p)).^2 + (Y - particles.y(p)).^2);

    % Regularize to avoid singularity
    r = max(r, epsilon);

    % Add contribution to potential
    phi = phi - particles.q(p) * log(r);
end

%% Compute electric field (negative gradient of potential)
[Ex, Ey] = gradient(-phi, dx, dy);
E_mag = sqrt(Ex.^2 + Ey.^2);

%% Visualization
figure('Position', [50, 200, 1400, 500]);

% Plot 1: Charge density
subplot(1, 3, 1);
imagesc(x, y, rho);
axis xy equal tight;
colorbar;
colormap(gca, bluewhitered(256));
hold on;
% Mark positive charges with red circles, negative with blue
pos_idx = particles.q > 0;
neg_idx = particles.q < 0;
scatter(particles.x(pos_idx), particles.y(pos_idx), 50*abs(particles.q(pos_idx)), 'r', 'filled', 'MarkerEdgeColor', 'k');
scatter(particles.x(neg_idx), particles.y(neg_idx), 50*abs(particles.q(neg_idx)), 'b', 'filled', 'MarkerEdgeColor', 'k');
title('Charge Density \rho');
xlabel('x'); ylabel('y');

% Plot 2: Electrostatic potential
subplot(1, 3, 2);
imagesc(x, y, phi);
axis xy equal tight;
colorbar;
colormap(gca, bluewhitered(256));
hold on;
contour(X, Y, phi, 20, 'k', 'LineWidth', 0.5);
scatter(particles.x(pos_idx), particles.y(pos_idx), 50*abs(particles.q(pos_idx)), 'r', 'filled', 'MarkerEdgeColor', 'k');
scatter(particles.x(neg_idx), particles.y(neg_idx), 50*abs(particles.q(neg_idx)), 'b', 'filled', 'MarkerEdgeColor', 'k');
title('Electrostatic Potential \phi');
xlabel('x'); ylabel('y');

% Plot 3: Electric field magnitude with streamlines
subplot(1, 3, 3);
imagesc(x, y, log10(E_mag + 1));
axis xy equal tight;
colorbar;
colormap(gca, hot(256));
hold on;
% Streamlines from positive charges
stream_density = 8;
startx = linspace(domain(1), domain(2), stream_density);
starty = linspace(domain(3), domain(4), stream_density);
[SX, SY] = meshgrid(startx, starty);
h = streamline(X, Y, Ex, Ey, SX(:), SY(:));
set(h, 'Color', [0.3, 0.3, 0.3], 'LineWidth', 0.5);
scatter(particles.x(pos_idx), particles.y(pos_idx), 50*abs(particles.q(pos_idx)), 'r', 'filled', 'MarkerEdgeColor', 'k');
scatter(particles.x(neg_idx), particles.y(neg_idx), 50*abs(particles.q(neg_idx)), 'b', 'filled', 'MarkerEdgeColor', 'k');
title('Electric Field |E| (log scale)');
xlabel('x'); ylabel('y');

sgtitle(sprintf('2D Electrostatic Field from %d Point Charges', n_particles));

%% Verify Poisson's equation: nabla^2 phi = -rho/epsilon_0
% Compute numerical Laplacian
laplacian_phi = del2(phi, dx, dy) * 4;  % del2 returns (1/4)*Laplacian

% For point charges, this should be approximately -rho (with delta functions)
% We can check away from particles
figure('Position', [50, 50, 700, 400]);

subplot(1, 2, 1);
imagesc(x, y, laplacian_phi);
axis xy equal tight;
colorbar;
colormap(gca, bluewhitered(256));
title('\nabla^2 \phi (numerical)');
xlabel('x'); ylabel('y');

subplot(1, 2, 2);
imagesc(x, y, -rho);
axis xy equal tight;
colorbar;
colormap(gca, bluewhitered(256));
title('-\rho (charge density)');
xlabel('x'); ylabel('y');

sgtitle('Verification: \nabla^2 \phi \approx -\rho');

%% Test superposition principle
% Generate two separate charge sets and verify phi(A+B) = phi(A) + phi(B)

fprintf('\n--- Superposition Test ---\n');

% Split particles into two groups
n_A = floor(n_particles / 2);
idx_A = 1:n_A;
idx_B = (n_A+1):n_particles;

% Compute potential for group A only
phi_A = zeros(ny, nx);
for p = idx_A
    r = sqrt((X - particles.x(p)).^2 + (Y - particles.y(p)).^2);
    r = max(r, epsilon);
    phi_A = phi_A - particles.q(p) * log(r);
end

% Compute potential for group B only
phi_B = zeros(ny, nx);
for p = idx_B
    r = sqrt((X - particles.x(p)).^2 + (Y - particles.y(p)).^2);
    r = max(r, epsilon);
    phi_B = phi_B - particles.q(p) * log(r);
end

% Check superposition
phi_sum = phi_A + phi_B;
superposition_error = max(abs(phi(:) - phi_sum(:)));
relative_error = superposition_error / max(abs(phi(:)));

fprintf('Max superposition error: %.2e\n', superposition_error);
fprintf('Relative error: %.2e%%\n', relative_error * 100);

if relative_error < 1e-10
    fprintf('Superposition test PASSED (machine precision)\n');
else
    fprintf('Superposition test FAILED\n');
end

%% Helper function for blue-white-red colormap
function cmap = bluewhitered(n)
    % Blue-white-red colormap centered at zero
    if nargin < 1
        n = 256;
    end

    half = floor(n/2);

    % Blue to white
    r1 = linspace(0, 1, half)';
    g1 = linspace(0, 1, half)';
    b1 = ones(half, 1);

    % White to red
    r2 = ones(n-half, 1);
    g2 = linspace(1, 0, n-half)';
    b2 = linspace(1, 0, n-half)';

    cmap = [r1, g1, b1; r2, g2, b2];
end
