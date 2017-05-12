% ACM 11: Introduction to MATLAB

%% Week 6: Optimization, Interpolation, Curve Fitting, Numerical Integration

%% Optimization problems

%{
A huge number of useful tasks can be accomplished by solving a problem of
this form:

Minimize f(x) for x in Omega
such that g_i(x) <=0  for i =1,...,m

This is called an (mathematical) optimization problem,
or sometimes a 'program'.

x is an element of R^n, f is the objective function, the g_i are the
constraint functions, and Omega (a subset of R^n) is the domain.

As you might imagine, this is a very general class of problem, and
imposing more structure makes it easier to solve. Some things that make
it easier:

-  m = 0 (the problem is 'unconstrained')

-  f and g_i are linear or quadratic

   Omega is "simple" (all of R^n, an intersection of half-spaces,
      a convex set,..)

-  f and g are convex (definition below)

An important strategy in solving an optimization problem is to write it in
the  simplest way possible. A simple yet very important example:

    minimize (log x)^2
    such that x > 0

is a nonlinear, constrained problem, yet a simple change of variables
y = log(x) lets us solve the quadratic, unconstrained problem

    minimize y^2

This may seem absolutely trivial, but this kind of insight can make
a massive difference to the time it takes to solve a problem!
%}
%% Optimization
% MATLAB comes with a useful set of optimization tools in the optimization
% toolbox (which is available with all of Caltech's MATLAB versions)

%% Root-Finding
%  Finding zeros of functions is very related to minimization.
%  Indeed, in the simplest version a root-finding method - Newton's method
%  is minimizing a function by finding a zero of its gradient
clc; clear; close all;
f = @(x) cos(x)+0.1*x;
ezplot(f,[-10,10]);
grid on;
x0=fzero(f,0.5) % Finds a point x where f(x)=0. x0=0.5 is the initial value.
hold on;
plot(x0,f(x0),'.r','MarkerSize',20)
% fzero rearches for the root locally:
x1=fzero(f,-6)
plot(x1,f(x1),'.g','MarkerSize',20)

% Important remarak: For fzero, "solution" is where f(x) changes sign.
% Lets try
f=@(x) x.^2;
x0=fzero(f,0.5)
% According to fzero, x0=0 is not a solution, since f does not change sign!
% Function fsolve resolves this problem:
x0=fsolve(f,0.5)  % It is more general, it solves systems of nonlinear equations

% Why do we need fzero then?
% Let us consider an example:
f=@(x) x.^3;
x1=fzero(f,0.5)
x2=fsolve(f,0.5)
% fzero's answer is correct to within machine precision!
% fsolve gives an approximate solution
% The goals of two functoin are different:
% - fzero:  solves accurately simple equations
% - fsolve: solves approximately compicated (systems of) equations.

f1=@(x) x(1).^2+x(2).^2-25;
f2=@(x) 3*sin(x(1))+4*cos(x(2));
F=@(x) [f1(x); f2(x)];
x = fsolve(F,[0;0])

figure
fsurf(@(x,y) f1([x,y]),'FaceColor','blue')
hold on
fsurf(@(x,y) f2([x,y]),'FaceColor','green')
plot3(x(1),x(2),f1(x),'or','MarkerSize',10,'MarkerFaceColor','red')

% As you can tell by the fact that you need an initial guess,
% the algorithms only search locally and return local root.

x1 = fsolve(F,[3;3])
plot3(x1(1),x1(2),f1(x),'om','MarkerSize',10,'MarkerFaceColor','magenta')

%% Unconstrained Minimization
clear; clc; close all;
f = @(x) 0.1*(x+1).^2 + cos(x);
ezplot(f);
hold on;
[xmin0, fval0] = fminunc(f,0); % finds minimum of unconstrained function
[xmin1, fval1] = fminunc(f,1);
plot([xmin0 xmin1],[fval0 fval1],'rx');

% Or a function of several variable:
g = @(x) (x(1)-1).^2 + (x(2)-1).^2;
x0 = [2,2];
[x,gmin] = fminunc(g,x0)

% 'Wrapping objective functions'

clc; clear; close all;
% It's often the case that the function you want to minimize
% has various parameters (extra arguments that you want to 'freeze')
f = @(x,param) x.^2 + param*(x-1).^2;

% f as written isn't in the form that fminunc expects:
fminunc(f,0)

% But we can 'wrap it'
paramVal = 0.3243;
f_wrapped = @(x) f(x,paramVal);

[xmin, fmin] = fminunc(f_wrapped,0)

%% Constrained Minimization

% Minimization of a single-variable function on fixed interval: fminbnd
close all; clear; clc
f = @(x) 4*cos(x) + (x-1).^2 + (x-2).^(2);
[xmin,fval] = fminbnd(f,0,4);
ezplot(f);
hold on;
plot(xmin,fval,'rx');

% Linear Programs
%{
x=linprog(f,A,b,Aeq,beq,lb,ub): finds the minimum of the following
linear problem:

minimize f'*x
s.t. A*x-b <= 0
     Aeq*x=beq
     lb<=x<=ub
%}

% Example:
close all; clear; clc
f = [-1; -1/3];
A=  [1 1; 1 1/4; 1 -1; -1/4 -1; -1 -1;  -1 1];
b = [2; 1; 2; 1; -1; 2];
lb = zeros(2,1);  %lower bounds
ub = 2*ones(2,1);% upper bounds

[x,fval,exitflag] = linprog(f,A,b,[],[],lb,ub)

% - x is the solution
% - fval is the value of the objective function at the solution
% - exitflag is a value that describes the reason why linprog stopped
%     1: Function converged to a solution x.
%     0,-2,-3,-4,-5,-7: "bad news", for details: doc linprog

% Linearly constrained Quadratic Programs
%{
x = quadprog(H,f,A,b,Aeq,beq,lb,ub): finds the minimum of the following
quadratic problem:

Minimize (1/2)*x'*H*x + f'*x
s.t. A*x-b <= 0
     Aeq*x=beq
     lb<=x<=ub

Very similar to linprog.
%}
clc; clear; close all;
H = [1 -1; -1 2];
f = [-2; -6];
A = [1 1; -1 2; 2 1];
b = [2; 2; 3];
lb = zeros(2,1);
[x,fval,exitflag] =  quadprog(H,f,A,b,[],[],lb,[])

% General constrained minimization (hard!)
%  We won't get into the generalities of this.
%  This function can handle all of the previous examples and more,
%  but many problems of this type are computationally intractable
doc fmincon

%% Convex Optimization
%{

If A and B are vectors in R^n, a convex combination of A and B is an
expression of the form

sA +(1-s)B

where s is a real number in [0,1]. The set of all convex combinations
of A and B is thus the line segment between A and B.

A set is convex if it contains all convex combinations of all of its
poitns. That is, you can 'see' any point in the set from any other point.

A function f:R^n -> R defines a set (its 'epigraph') of all the points in
R^(n+1) with x_(n+1) >= f(x_1,...,x_n). That is, the set 'above' the
function.

A function is called convex if its epigraph is a convex set. The key
property of convex functions is that LOCAL minimizers are in fact GLOBAL.

A convex optimization problem is one in which the objective function, the
domain, and the constraints are all convex.

The conventional wisdom is that Matlab is very good at numerical linear
algebra and related operations, but is relatively poor at optimization.
But there are many excellent 3rd party Matlab applications, some of them
free, that do optimization very well.  A particularly simple package is
"cvx" by Michael Grant and Stephen Boyd.

Check:
 - This (free, excellent) textbook: http://www.stanford.edu/~boyd/cvxbook/
 - The CVX manual: http://web.cvxr.com/cvx/doc/CVX.pdf

This package is not always the (computationally) fastest way to solve a
problem, especially a large one, but it is extremely useful for
small-to-medium convex programming, and may well be the fastest approach
in terms of your own time.
%}

%% Interpolation
% MATLAB offers several function to perform 1d interpolation.
% We consider: interp1 (other: interpft, spline, pchip, etc)
% Note: for all interpolations, the values in x must be monotonic
% (increasing or decreasing) - use the sort function.
clc; clear; close all;
x = [0 0.785 1.570 2.356 3.141 3.927 4.712 5.497 2*pi];
y = [0 0.707 1.000 0.707 0.000 -0.707 -1.000 -0.707 -0.000];

% use plot tools
figure
subplot(2,2,1)
plot(x,y,'ob');
hold on
plot(x,y,'-r');  % linear interpolation
title('Using plot');

% Using interp1
xi = linspace(0,2*pi,1000); % points to interpolate value at
yinearest = interp1(x,y,xi,'nearest'); % nearest neighbor interpolation
yilinear = interp1(x,y,xi,'linear');
yispline = interp1(x,y,xi,'spline');
% To read about different methods: doc interp1

subplot(2,2,2)
plot(x,y,'o',xi, yinearest);
title('Nearest neighbor interpolation');
subplot(2,2,3)
plot(x,y,'o', xi, yilinear);
title('Linear interpolation');
subplot(2,2,4)
plot(x,y,'o',xi, yispline);
title('Spline interpolation');

%% Polynomial Fitting
clc; clear; close all;
% Using the GUI (Tools->Basic Fitting)
x=sort(randn(1,5));
y=randn(1,5)
plot(x,y,'o','MarkerFace','blue')

% MATLAB represents polynomials by vectors of coefficients
P = [1 0 2 3]; % corresponds to p(x) =  x^3 + 2*x + 3
% Evaluating with polyval:
% y = polyval(p,x) returns the value of a polynomial defined by p evaluated
% at x. The input argument p is a vector of length n+1 whose elements are
% the coefficients in descending powers of the polynomial to be evaluated.
xx = linspace(-3,3,100);
figure
plot(xx,polyval(P,xx),'.');

% Fitting with polyfit:
% p = polyfit(x,y,n) returns the coefficients for a polynomial p(x) of
% degree n that is a best fit (in a least-squares sense) for the data in y.
Pfit = polyfit(x,y,4);
yfit = polyval(Pfit,[min(x):0.01:max(x)]);
figure
plot(x,y,'o','MarkerFace','blue')
hold on
plot([min(x):0.01:max(x)],yfit,'k-');

%% Fitting General Functions

% Linear Least Squares
close all; clear; clc

f = @(beta,x) beta(1)*cos(x) + beta(2)*exp(x); %linear in the parameters!
n = 1000;             % number of data points
x = linspace(0,1,n)'; % "inputs"
betaRef = [1 -1];     % true value of beta
y = f(betaRef,x) + 0.1*randn(n,1);  % "outputs", the 2nd term is noise
A = [cos(x) exp(x)];
betaEst = A\y;     % solves data=A*beta for beta
figure
hold on
plot(x,y,'r+')
plot(x,f(betaRef,x),'g-','LineWidth',2)
plot(x,f(betaEst,x),'b-.','LineWidth',1)
legend('data','ground truth','fit');

% Nonlinear curve fitting
close all; clear; clc;
f = @(beta,x) beta(1) + exp(beta(2)*x) + sin(beta(3)*x); % nonlinear in parameters
n = 1000;
x = linspace(0,pi,n)';
betaRef = [1 -1 1];
y = f(betaRef,x) + 0.1*randn(n,1);
% lsqcurvefit: solves nonlinear curve-fitting problems in least-squares sense
% beta = lsqcurvefit(f,beta0,x,y) starts at beta0 and finds parameters beta
% to best fit the nonlinear function f(beta,x) to the data y
% (in the least-squares sense).
% In other words, it solves: ||f(beta,x) - y ||_2 -> min

beta0 = [0.5,0.5,0.5];
% beta0 = [2,2,2];
betaEst = lsqcurvefit(f, beta0, x, y)
figure
hold on
plot(x,y,'r+')
plot(x,f(betaRef,x),'g-','LineWidth',2)
plot(x,f(betaEst,x),'b-.','LineWidth',1)
legend('data','ground truth','fit');

%% Making nonlinear porblem linear is not always a good idea!
%{
close all; clear; clc
f = @(beta,x) beta(1)*exp(beta(2)*x); %nonlinear
n = 1000;
x = linspace(0,1,n)';
betaRef = [1 1];
y = f(betaRef,x) + 0.1*randn(n,1);

% nonlinear way:
beta0 = [3, -3];
betaEst1 = lsqcurvefit(f, beta0, x, y);

% converting to a linear fitting problem
% take the log: log(y) = m(1) + m(2)*x, where
% m(1)=log(beta(1)) abd m(2)=beta(2)
logData = log(y);
A = [ones(length(x),1), x];
mEst = A\logData;
betaEst2 = [exp(mEst(1)), mEst(2)];
figure
hold on
plot(x,y,'r+')
plot(x,f(betaRef,x),'r-','LineWidth',1)
plot(x,f(betaEst1,x),'b--','LineWidth',1)
plot(x,f(betaEst2,x),'m--','LineWidth',1)
legend('data','ground truth','"nonlinear" fit','"linear" fit');
%}
%% Numerical Integration (aka quadrature)
%{
Numerical integration is the process of approximating the value of an
integral.  The general technique, called quadrature, is to sample the
function f at a few points, interpolate these points piece-wise with a
polynomial of degree n, integrate this polynomial exactly, and return this
as our approximation.  The interpolation and exact integration can be
combined into one step, so no actual interpolation really happens.
%}

%{
Matlab choses gridpoints adaptively. The workhorse routine is "integral":
-   I = integral(f,xmin,xmax) numerically integrates function f
from xmin to xmax using using global adaptive quadrature and default
error tolerances.
-   I=integral(fun,xmin,xmax,Name,Value)  specifies additional options
with one or more Name,Value pair arguments, e.g.
'AbsTol',1e-12 sets the absolute error tolerance to approximately 12
decimal places of accuracy.
%}

clc; clear; close all;
f=@(x) exp(-(x.^2)/2);    % the integrand
Iapprox = integral(f,-Inf,Inf)
Iexact = sqrt(2*pi)

% absolute error
error=abs(Iexact - Iapprox)

% MATLAB can also do double and triple integration
doc integral2
doc integral3

%% Dialog Boxes and Waitbar
% Matlab can popup dialog boxes (see "dialog" or "msgbox").
clc; clear; close all;
msgbox('Mission Accomplished');

% Here's an example of a waitbar: it graphically displays a sliding counter.
h = waitbar(0,'Computing...');
for i = 1:300
    pause(.05);         % wait .05 seconds
    % (usually, a lengthy computation goes here)
    waitbar(i/300,h);
end
close(h);

%% Remember: Matlab always has the answer.
% If puzzled, just type and execute in the command line:
why
