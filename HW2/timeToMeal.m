function tDevour = timeToMeal(alpha)
% TIMETOMEAL Calculate the time before you are devoured by raptors in
% seconds assuming that you start running at angle alpha and raptors 1 and
% 2 are at (-10, -a) and (10, -a) with velocities of 25 m/s and raptor 3 is
% at (0, 2a) with a velocity of 20 m/s

%% timeToMeal(30) = 0.4265

% Angle in degrees
alphaDeg = deg2rad(alpha);

% Raptop velocities in m/s
v1 = 25;
v2 = 25;
v3 = 20;

% Your velocity in m/s
vh = 6;

% Your starting position and direction
initH = [0; 0];
e = [cos(alphaDeg); sin(alphaDeg)];

% Raptors are in d = 20 sized equilateral triangle
d = 20;
a = d * sqrt(3) / 6;
% Raptor starting positions
initR1 = [-10; -1 * a];
initR2 = [10; -1 * a];
initR3 = [0; 2 * a];

% Derivative function
% Let Y = h - r = [x = hx - rx; y = hy - ry]
% Y' = [hx' - rx'; y = hy' - ry']
% Y' = h' - vi * [h - r] / ||h - r||
% Y' = h' - vi * [Y] / ||Y||

% h' = vh t e
% r' = vi (h - r) / ||h - r||
% x' = hx' - rx'
% y' = hy' - ry'
h = @(t) vh * t .* e;
dhdt = vh .* e;
opt = odeset('Events', @caughtEvents);
tspan = [0 1];

% Solve for when raptor 1 would eat you
odefun1 = @(t, Y) [dhdt - v1 * ([Y(1); Y(2)]) / norm([Y(1); Y(2)])];
initY1 = [initH - initR1];
[~, ~, t1, ~, ~] = ode45(odefun1, tspan, initY1, opt);

% Solve for when raptor 2 would eat you
odefun2 = @(t, Y) [dhdt - v2 * ([Y(1); Y(2)]) / norm([Y(1); Y(2)])];
initY2 = [initH - initR2];
[~, ~, t2, ~, ~] = ode45(odefun2, tspan, initY2, opt);

% Solve for when raptor 3 would eat you
odefun3 = @(t, Y) [dhdt - v3 * ([Y(1); Y(2)]) / norm([Y(1); Y(2)])];
initY3 = [initH - initR3];
[~, ~, t3, ~, ~] = ode45(odefun3, tspan, initY3, opt);

% Solve for first raptor to eat you
tDevour = min([t1, t2, t3]);
end
