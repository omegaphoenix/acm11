% EXAMPLE This script uses the discreteLaplace function to compute an
% approximation to delta u on the unit square, with ny=30 and nx=20
% and u(x,y) = cosh(2pix)cos(6piy), x in [.25, .75] and y in [.25, .75]
% and u(x,y) = 0 otherwise.

ny = 30;
nx = 20;
hy = 1 / (ny - 1);
hx = 1 / (nx - 1);

% Create x, y grid of coordinates to plot
rx = linspace(0, 1, nx);
ry = linspace(0, 1, ny);
[x,y] = meshgrid(rx, ry);

% Create u(x,y) = cosh(2pix)cos(6piy), x in [.25, .75] and y in [.25, .75]
% and u(x,y) = 0 otherwise.
u = (x >= .25 & x <= .75 & y >= .25 & y <= .75)...
    .* cosh(2 * pi * x) .* cos(6 * pi * y);

% Apply Laplace operator
delta_u = discreteLaplace(ny, nx, hy, hx) * u(:);
delta_u = reshape(delta_u, ny, nx);

% Create figure
figure

% Plot U on the left side
subplot(1, 2, 1);
surf(x, y, u)
xlabel('x');
ylabel('y');
zlabel('u');
title('u');

% Plot delta U on the right side
subplot(1, 2, 2);
surf(x, y, delta_u);
xlabel('x');
ylabel('y');
zlabel('u');
title('\Delta u')