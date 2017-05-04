% ACM 11: Introduction to MATLAB

%% Week 4: Norms, Probability, Statistics, Ordinary Differential Equations
%%         Optimization of a single-variable function on a fixed interval

%% Norms
% Key point for all norms:  norm(x) = 0 if and only if x = 0
clc; clear; close all;
x = randn(20,1);
A = randn(20,20);
disp('Vector norms:')
norm(x)         % if given no second argument, the default is "2"
norm(x,2)       % p-norm norm(x,p)=(sum |x_i|^p)^(1/p)
norm(x,1)
norm(x,'inf')
disp('Matrix norms:')
% ||x||_p = (\Sum |x_i|^p)^(1/p), p >= 1
% As p -> Inf, ||x||_p -> ||x|| = max{|x_1|,...,|x_n|}
%||A||_p = sup_{x = 0} \frac{||A_x||_p}{||x||_p}
%||A||_F = sqrt( \Sum_{i, j} |a_{ij}|^2
norm(A)         % if given no second argument, the default is "2"
norm(A,2)       % p-norm: sup norm(A*x,p)/norm(x,p)
norm(A,'fro')   % Frobenious norm = (sum |a_ij|^2)^(1/2)

%% Basic Probability and Statistics
clc; clear; close all;
% Random numbers
% Random number generator
% Fix a number c_0, called "seed"
% c_k = (A * c_{k-1}) mod M where A is the multiplier and M is a large prime number
% M = 2^x - 1
% u_k = c_k / M \in [0, 1 - 1/M] = [0, 1]
% u_1, ..., u_k ~ U[0,1]
m=3; n=2;
rand(m,n)   % uniform [0,1]
randn(m,n)  % N(0,1)
imax=100;
r = randi(imax,n) % randi(imax,n) returns an n-by-n matrix of pseudorandom
                  % integers drawn from the discrete uniform distribution
                  % on the interval [1,imax].

% To control random number generation (useful for reproducibility), use rng.
% rng(seed) seeds the random number generator using the nonnegative integer
% seed so that rand, randi, and randn produce a predictable sequence of numbers
rng(2017)
x=rand(1,5)
rng(2017)
y=rand(1,5)
rng('default') % puts the settings of the random number generator used by
               % rand, randi, and randn to their default values.
               % This way, the same random numbers are produced as if you
               % restarted MATLAB.
s=rng;   % saves the current generator settings in s
x=rand(1,5)
% Now you can give s to your frined, and s/he will be able to reproduce x:
rng(s);  % Restores the original generator settings in s
y=rand(1,5)

%% Distributions
clc; clear;
doc random % VAST library of random distributions
x = random('Poisson',10,[2,3]) % generates a 2*3 matrix, where each entry
                               % is a realization of Poisson rv with mean 10
y = random('Lognormal',.1,.4,[1,6]) % generates 6 lognormally distributed
% random variables with mu=0.1 and sigma=0.4.

% y = pdf('name',x,A) returns the probability density function (pdf) for
% the one-parameter distribution family specified by 'name', evaluated at
% the values in x. A contains the parameter value for the distribution.
f=pdf('Normal',[-3:0.1:3],0,1);
plot([-3:0.1:3],f)
% cdf returns the cumulative distribution functions CDF
F=cdf('Normal',[-3:0.1:3],0,1);
plot([-3:0.1:3],F)

% Yet another example
x = 0:.01:10;
dof = 3;  % degrees of freedom
f = pdf('Chi2',x,dof);
F = cdf('Chi2',x,dof);
plot(x,f)
hold on;
plot(x,F,'r');
legend('PDF','CDF')

%% Maximum Likelihood Estimates
% The method is explained on the board
% x_1 ... x_n ~ f(x | \theta) where \theta are the parameters
% Likelihood function L(\theta) = f(x_1,...x_n | \theta)
% Joint distribution of x_1,...,x_n = \Prod_{i=1}^k f(x_i | \theta)
clc; clear;
% Generate 100 random observations from a binomial distribution
% with the number of trials, n = 20, and the probability
% of success, p = 0.75.
data = binornd(20,0.75,100,1);

% Find the MLE of p:
p_mle = mle(data,'distribution','binomial','ntrials',20)

%% Monte Carlo Method for Integration
% The method is explained on the board
% 1. X = \Pi(x) is the PDF
%    E[f(x)] = \Int_{-Inf}^{Inf} f(x) \Pi(x)dx
% 2. Law of Large Numbers
%    E_{\pi} [f(x)] \approx \frac{1}{N} \Sum_{i=1}^N f(x_i);
%    x_1 ... x_N ~ \Pi(x)
% Monte carlo Method (for integration)
% 1. I = \Integral_a^b f(x)dx = \frac{b-1}{N} \Sigma_{i=1}^n f_i(x)
% 2. I = \Int ... \Int_V f(x) dx = Volume(V) \Int ... \Int_V \frac{f(x)}{volume(V)} dx
%      = Volume(V) E_{U[V]} [f(x)]
%      \a[[rox \frac{Volume(V)}{N} \Integral_{i=1}^N f(x_i)
% Accuracy does not depend on dimension of V
% Might not be trivial to sample from uniform distribution on V, U[V]
% Trick (only works in relatively low dimensions)
% I = \Int ... \Int f(x) I_V(x) dx where I_V(x) is the indicator function for V
%   \approx \frac{Volume(V')}{N} \Sum_{i=1}^N f(x_i) I_V(x_i)

% Volume between an ellipsoid (a,b,c) and sphere with radius r
clear;clc;close all;
r=5; a=2; b=1; c=.8;
N=100000; % Number of Monte Carlo samples
x = -r +2*r*rand(N,1);
y = -r +2*r*rand(N,1);
z = -r +2*r*rand(N,1);
between = (x.^2+y.^2+z.^2<=r^2) & (x.^2./a^2 + y.^2./b^2+z.^2./c.^2>=1);
% between(i)=1  <=> (x(i),y(i),z(i)) is inside sphere but outside the
% ellipsoid
area = (2*r)^3*sum(between)/N % (2*r)^3 is the volume of the cube
area_exact =  4/3*pi*r.^3 - 4/3*pi*a*b*c

%% ODEs: Built-in Examples

% MATLAB contains several very nice built-in example.
% Examine some of them by executing:
odeexamples

%% MATLAB's ODE Solvers for initial-value problems

% MATLAB  has first-order, numerical ODE solvers for initial value problems
% In general, use them as follows: [t,Y] = solver(odefun,tspan,y0)
% to solve dY/dt = odefun(t,Y), Y(0) == y0 for t = tspan(1) to tspan(end)

% Use [t_0:\delta t:t_1] instead of [t_0,t_1] if you know what you are doing

% Example 1: y' = -y , y(0)=1, on [0,1];
close all; clear; clc
tspan = [0, 1];  % Matlab will automatically descritize the interval
%tspan = linspace(0,1,100); %uncomment to compute at these points
odefun = @(t,y) -y;
y0 = 1;
[t,y] = ode45(odefun, tspan, y0); % ode45 is one of many ODE solvers
                                  % Matlab recommens it as the first
                                  % solver you should try.
odeExactSol = @(t) y0 * exp(-t);
figure
plot(t,y,'b-x',t,odeExactSol(t),'r-o');
legend('Approx','Exact');

% Strategy for solving higher order ODE y^(k)=F(y^(k-1),...,y^(2),y',t):
% convert to first-order system of ODEs by introducing new variables,
% one for each derivative y', y^(2), ... y^(k-1).

% Example 2:
%{
 y'' + y' + y = 0, y(0) = 1, y'(0) = 0  on [0,10]

 let z = y'(t) then the (first order) system is

 y' = z
 z' = -z - y
 y(0) = 1
 z(0) = 0

 let Y = [y ; z]

%}
close all; clear; clc
odefun = @(t,Y) [Y(2);-Y(2)- Y(1)];
Y0 = [1, 0]';
tspan = [0, 10];
%tspan = linspace(0,10,1000);
[t,Y] = ode45(odefun,tspan,Y0);
odeExactSol = @(t) 1/3*exp(-t/2).*(3*cos(sqrt(3)*t/2) +...
    sqrt(3)*sin(sqrt(3)*t/2));
figure
subplot(2,1,1);
plot(t,Y(:,1),'b-x',t,odeExactSol(t),'r-o');
legend('Approx','Exact');
subplot(2,1,2);
plot(t,abs(Y(:,1)-odeExactSol(t)),'bx');
legend('Error');

% Example 3:
%{
Now consider y as a 2-vector, and A,B,C as 2*2 matrices

Ay'' + By' + Cy = 0
y(0)  = [1 1]';
y'(0) = [0 0]';

Solve for the highest derivative:
y'' + inv(A)*B y' +inv(A)*C*y

Let z=y', then
y'=z
z'=-inv(A)*B*z-inv(A)*C*y
y(0) = [1 1]'
z(0)= [0 0]'
Let let Y = [y ; z]
%}
close all; clear; clc
A = 1+rand(2);
B = 1+rand(2);
C = 1+rand(2);
odefun = @(t,Y) [Y(3:4); -inv(A)*(B*Y(3:4)+C*Y(1:2))];
Y0 = [1 1 0 0]';
tspan = [0 10];
[t,Y] = ode45(odefun,tspan,Y0);
figure
plot(t,Y(:,1:2));
legend('Y_1','Y_2');

%% Choosing a Solver
% There is no precise answer, but here is a rule of thumb from MATLAB:
%         Accuracy        When to Use
% ode45:  Medium          Most of the time.
% ode23:  Low             We you need a crude solution
% ode113: Low to high     When function evals are expensive

close all; clear; clc
% Let us consider the following 1st order ODE:
% y'=cos(t)-y, y(0)=0, on [0,5]
odefun = @(t,y) cos(t) - y;
tspan = [0 5];
y0 = 0;
exact = @(t) 0.5*(-exp(-t) + cos(t) + sin(t));

% The first thing to try, ode45 - the workhorse
tic
[t,y]=ode45(odefun, tspan, y0);
comptime = toc;
subplot(2,4,1);
plot(t, y, t, exact(t), 'r--')
title(sprintf('ode45 [%f s]',comptime));
subplot(2,4,5);
plot(t,abs(y-exact(t)),'bx');
title('Error');

% If you need only a crude solution
tic
[t, y] = ode23(odefun, tspan, y0);
comptime = toc;
subplot(2,4,2);
plot(t, y, t, exact(t), 'r--')
title(sprintf('ode23 [%f s]',comptime));
subplot(2,4,6);
plot(t,abs(y-exact(t)),'bx');
title('Error');

% When function evals are expensive
tic
[t, y] = ode113(odefun, tspan, y0);
subplot(2,4,3);
comptime = toc;
plot(t, y, t, exact(t), 'r--')
title(sprintf('ode113 [%f s]',comptime));
subplot(2,4,7);
plot(t,abs(y-exact(t)),'bx');
title('Error');

% Note that all of these are not super accurate!
% The default behavior is to go for speed, not accuracy.

% If you need accuracy:
tic
opt=odeset('RelTol',1e-10,'AbsTol',1e-10,'NormControl','on','MaxStep',0.001);
[t,y] = ode45(odefun, tspan, y0, opt);
comptime = toc;
subplot(2,4,4);
plot(t, y, t, exact(t), 'r--')
title(sprintf('ode45 again [%f s]',comptime));
subplot(2,4,8);
plot(t,abs(y-exact(t)),'bx');
title('Error');

%% Stiff ODEs
%{
  It is difficult to formulate a precise definition of stiffness,
  but the main idea is that the ODE includes some terms that can lead
  to rapid variation in the solution. For example, the solution may
  vary slowly, but then very fast, and then slowly again.

  For a stiff ODE, certain numerical methods for solving the equation are
  numerically unstable, unless the step size is taken to be extremely small.

  Stiffness is an efficiency issue.
  If we weren't concerned with how much time a computation takes,
  we wouldn't be concerned about stiffness. Nonstiff methods can
  solve stiff problems; they just take a long time to do it.

  A model of flame propagation provides an example.
  When you light a match, the ball of flame grows rapidly until it
  reaches a critical size. Then it remains at that size because the
  amount of oxygen being consumed by the combustion in the interior
  of the ball balances the amount available through the surface.
  The simple model is

  y' = y^2- y^3,  y(t) is the radius of the ball
  y(0) = delta         delta is the (small) initial radius
  0 < t < 2/delta
%}
close all; clear; clc
odefun = @(t,y) y^2-y^3;
delta=0.01;  % not very stiff
y0 = delta;
tspan = [0 2/delta];
tic
[t,y] = ode45(odefun,tspan,y0);
comptime = toc;
subplot(2,2,1)
plot(t,y);
title(sprintf('ode45, delta=%f: %d steps and %f seconds\n',...
    delta, length(t),comptime));

delta=0.0001;         % quite stiff
tspan = [0 2/delta];
y0 = delta;
tic
[t,y] = ode45(odefun,tspan,y0);
comptime = toc;
subplot(2,2,2)
plot(t,y);
title(sprintf('ode45, delta=%f: %d steps and %f seconds\n',delta,...
    length(t),comptime));

% Now let us use ode15s, a stiff solver
tic
[t,y] = ode15s(odefun,tspan,y0);
comptime = toc;
subplot(2,2,4)
plot(t,y);
title(sprintf('ode15s, delta=%f: %d steps and %f seconds\n',delta,...
    length(t),comptime));

delta=0.01;         % back to not so stiff
tspan = [0 2/delta];
y0 = delta;
tic
[t,y] = ode15s(odefun,tspan,y0);
comptime = toc;
subplot(2,2,3)
plot(t,y);
title(sprintf('ode15s, delta=%f: %d steps and %f seconds\n',delta,...
    length(t),comptime));

% Others solvers:
% ode23s
% ode23t
% ode23tb

%% Implicit ODEs

% [t,Y] = ode15i(odefun,tspan,y0,yp0) solves implicit differential equations
% - odefun is a function (or function handle) that evaluates the left side
%          of the ODE, which are of the form f(t,y,y') = 0.
% - y0,yp0 are vectors of initial conditions for y and y' respectively.
% Simple example:
% y'*y+y=0, y(0)=1, y'(0)=-1
close all; clear; clc
odefun = @(t,y,yp) yp.*y + y;
[t,y] = ode15i(odefun,[0 1],1,-1);
plot(t,y);

%% Events
% In some ODE problems the times of specific events are important,
% such as the time at which a ball hits the ground, or the time at which
% a spaceship returns to the earth. While solving a problem, the ODE
% solvers can detect such events by locating transitions to, from, or
% through zeros of user-defined functions.

% Example 1: Throwing a Ball
% Let Y be of the form [x y x' y']'
close all; clear; clc;
odefun = @(t,Y) [Y(3); Y(4) ; 0; -9.8];
Y0 = [0.5 ; 0.5; 0.1; 0];  % throwing a ball from (0.5,0.5) with horozontal
                           % speed 0.1.
tspan = [0,10];
[t,Y] = ode45(odefun,tspan,Y0);
plot(Y(:,1),Y(:,2));  % makes no physical sense: the ball bounce off the,
                      % ground, it does not go under the ground!

% opt=odeset('Events',@events),
% where @events is a handle to a function of the following form:
% [value,isterminal,direction] = events(t,y)
% Event = moment in time when some function F(y(t))=0
% value, isterminal, and direction are vectors:
% - value(i) is the value of the ith event function (F(y(t))).
%            ith event occurs when value(i)=0
% - isterminal(i) describes "what to do" when ith event occurs:
%    = 1 if you want to stop the integration.
%    = 0 otherwise.
% - direction(i)
%    = 0  if all zeros are to be located,
%    = +1 if only zeros where the event function is increasing, and
%    = -1 if only zeros where the event function is decreasing.

opt = odeset('Events',@bounceEvents);
numBounces = 10;
figure; hold on; axis([0,1.5,0,1])
for i=1:numBounces
    [~,Y] = ode45(odefun,tspan,Y0,opt);
    plot(Y(:,1),Y(:,2));
    pause(.5)
    drawnow
    Y0 = Y(end,:)';
    Y0(4) = -Y0(4)*.9; %reverse y velocity
end
hold off

% Example 2: Bouncing within a box
Y0 = [0 ;1; 1; 0];  % initial conditiona
opt = odeset('Events',@bounceEvents2);
tspan = [0:0.01:50];
numBounces = 50;
figure; hold on; axis([0,1,0,1])
for i=1:numBounces
    [~,Y,~,~,IE] = ode45(odefun,tspan,Y0,opt);
    %[T,Y,TE,YE,IE] = solver(odefun,tspan,y0,options)
    % TE: the time at which an event occurs.
    % YE: the solution at the time of the event.
    % IE: the index i of the event function that vanishes
    plot(Y(:,1),Y(:,2),'b');
    pause(0.5)
    drawnow
    Y0 = Y(end,:)';
    if(IE == 1) % the ball hits the ground
        Y0(4) = -Y0(4); %reverse y velocity
    else % the ball hits wall
        Y0(3) = -Y0(3); % reverse x velocity
    end
    Y0(3:4) = 0.95*Y0(3:4); % reduce velocities
end
hold off

%% Boundary Value Problems

% MATLAB has similar functions for solving boundary value problems

doc bvp4c
doc bvp5c
doc bvpset

%% Optimization (more on Optimization in Week 6)

% Minimization of a single-variable function on fixed interval: fminbnd
close all; clear; clc
f = @(x) 4*cos(x) + (x-1).^2 + (x-2).^2;
[xmin,fval] = fminbnd(f,0,4); % find min of f on [0,4]
ezplot(f,[0,4]);
hold on;
plot(xmin,fval,'rx');
