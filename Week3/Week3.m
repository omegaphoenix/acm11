% ACM 11: Introduction to MATLAB   

%% Week 3: 2D and 3D Graphics, Images, and Linear Algebra

%% The fplot function

% The easiest commands for plotting graphs of functins are
% - fplot and 
% - various ezplots
%
% fplot(function, [xmin xmax]) plots a function of a single variable in the
% specified range
% Here "function" is a) anonymous function or b) a function handle
clc; clear; close all;
figure('Position',[500 200 650 600]);
f=@(x) exp(-0.1*x).*sin(x);
fplot(f, [0,20])

% Here is an example with function handle
fplot(@sin, [0,20])

%% The ezplot family 
% ezplot (easy-to-plot) is more general.
ezplot(f,[0,20]) % works similarly to fplot, except it labels your graph 
                 % and x-axis. The default x range is [-2*pi, 2*pi]

% You can also plot implicitly defined functions F(x,y) = 0 using ezplot
F=@(x,y) x.^2+y.^2 - x.*y - 3;
ezplot(F, [-3,3]); % the limits [a,b] apply to both x and y 
                   % but you can also specify [xmin xmax ymin ymax]. 

% Parametric ezplot
r = @(t) exp(cos(t)) - 2*cos(4*t) - sin(t/12).^5; 
x = @(t) sin(t).*r(t);
y = @(t) cos(t).*r(t);
ezplot(x, y, [-10,10])    % Note the low quality 
                          % We can't specify the number of points to use :(

% More control with plot
figure('Position',[500 200 650 600]);
t = linspace(-10,10,5000);
plot(x(t),y(t))

% There are other ez plotters, e.g. ezpolar, ezcontour, ezcontourf
r=@(theta) sqrt(abs(cos(2*theta)));
ezpolar(r);  %plots the polar curve r= r(theta) over  0 < theta < 2pi
% ezpolar(r,[theta1,theta2]) if [theta1,theta2] is needed.

f=@(x,y) cos(x).*cos(y).*exp(-sqrt((x.^2+y.^2)/4)); 
ezcontour(f,[-7,7]); % plots the contour lines of f(x,y)

figure('Position',[500 200 650 600]);
ezcontourf(f,[-7,7])

% There are also ez plotters for 3d graphs: in a moment

% Subplot: allows to have several graphs in a single plot. 
close all 
figure('Position',[500 200 650 600]);
for row = 1:2       
    for col = 1:2   
        s=(row-1)*2+col;
        subplot(2,2,s); % work on the (r-1)*2+c plot in a 2x2 grid
        switch s
            case 1
                ezpolar(r);
            case 2
                ezcontour(f,[-7,7]);
            case 3            
                ezcontourf(f,[-7,7])
            otherwise,
                plot(x(t),y(t));
        end
    end
end

%% Examples of other useful common plotting commands 

close all;clear;clc;

% loglog
figure('Position',[500 200 650 600]);
x = linspace(0, 2*pi, 200);
y = 100*x.^3;
plot(x,y);
loglog(x,y);  grid on; % log-log scale plot

% bar creates a bar graph
% Probability example (explained on the board): 
p = linspace(0, 1, 200);
L1 = p.^2.*(1-p).^4;
L2 = p.^2.*(1-p).^6;
L3 = p.^5.*(1-p).^5;
bar(p,[L1',L2',L3']); % draws bars at the locations specified by p.
title('likelihood functions for three different datasets')
legend('2 heads, 4 tails', '2 heads, 6 tails', '5 heads, 5 tails');

% barh plots a bar graph horizontally
barh(p,[L1',L2',L3']);

% stem plots the data sequence(s)
x=linspace(0,2*pi);
y=[cos(x)',sin(x)'];
stem(y);

% A common scenario: we want to combine two variables on one plot, 
% but the variables have different scales.  
% Matlab provides the "plotyy" command for this
clear;clc; close all;
% Example of two variables with different scales
t = 1949:2017;
n=length(t);
stockMarket = 500*(randn(1,n) + 0.1*(t-1949));
population = exp(0.1*(t-1949));
figure('Position',[500 200 650 600]);
plot(t,stockMarket)
hold on
plot(t,population)
plotyy(t,stockMarket,t,population)

% Now, let's be a bit more advanced. We'll put the stock market data
% on a normal plot, but plot the world population on a log plot, since we
% know the growth is exponential.
[ax,h1,h2] = plotyy(t,stockMarket,t,population,@plot,@semilogy);
% ax is a handle for axes, h1 and h2 are handles for curves. 
% Adding labels: 
xlabel('year');
ylabel(ax(1), 'Stock Market Data');
ylabel(ax(2), 'World Population')
% What if we want to add markers and adjust the line thickness? 
set(h1,'marker','+')
set(h2,'linewidth',2)
legend([h1,h2],'Stock Market','Population','location','best')
title('A fancier graph!')

% To explore: area, histogram, stairs, compass, comet, contour, quiver, pcolor  ...

%% 3D Plotting

close all;clear;clc;
figure('Position',[500 200 650 600]);

% plot3: by default it connects neighboring points by lines. 
X=rand(1,100);
Y=rand(1,100);
Z=rand(1,100);
plot3(X,Y,Z);
% If you want just a cloud of points without connecting lines, use:
plot3(X,Y,Z,'o')

% To plot a surface, first of all, we need to create a grid: 
% [x,y]=meshgrid(xgr,ygr) replicates the grid vectors xgr and ygr 
% to produce a full grid. The coordinates of a point (i,j) on a grid are
% x(i,j) and y(i,j). 
clear;
r = linspace(0,2,100); 
[x,y]=meshgrid(r,r);
z = cos(x).*sin(2*y);

% surf(x,y,z) uses z for the surface height. 
% x and y matrices defining the x and y components of a surface.
surf(x,y,z); 
shading interp % interpolate between mesh values in the plot

% mesh  draws a wireframe mesh
mesh(x,y,z); 

% ezplot3 can be used for ploting a curve in 3D, described parametrically. 
x=@(t) cos(t);
y=@(t) sin(3*t);
z=@(t) cos(5*t);
ezplot3(x, y, z, [0,10]);
% ezsurf can plot surfaces of functions of two variables. 
F=@(x,y) cos(x).*sin(2*y);
ezsurf(F,[0 10 0 10])

%% Saving / Loading figures
close all;clear;clc;

% Once you've created a figure, you may print it, export it to file, or
% save it for later loading in Matlab

% fig files
figure('Position',[500 200 650 600]);
plot(1,1,'*r','MarkerSize',10);
hgsave(gcf, 'FigStar'); % creates a .fig file, which is matlab's own format
% hgsave('filename') saves the current figure to a file named filename.
% hgsave(h,'filename') saves the figure identified by handle h to a file 
% named filename.

close all
% to load the figure: 
hgload FigStar

% If you want to export to other file formats (useful for papers), 
% use the menu in the figure window, or use saveas, e.g.
saveas(gcf, 'FigStar.eps')
saveas(gcf, 'FigStar.pdf')

% In general, you can use the figure window to manipulate all the
% parameters of your graphs after they've been created.

% A useful feature is the "Generate Code..." in the File menu. 
% This lets you play with the GUI and then obtain code to produce 
% a similar style

%% Images

close all;clear;clc;

% Images are represented as matrices in MATLAB. 
% There are two types: Indexed and true RGB (explained on the board)

% Indexed (aka colormapped) images are 2-dim matrices A with integer entries. 
% Each entry A(i,j) is an index into the colormap C. 
% Colormap C is a N-by-3 matrices, where each row C(i,:) defines a color in
% the RGB space: 
% - C(:,1) are intensities of Red (varying from 0 to 1), 
% - C(:,2) are intensities of Green (varying from 0 to 1), 
% - C(:,3) are intensities of Blue (varying from 0 to 1).
% Thus, the color of pixel (i,j) is C(A(i,j),:).
% [0 0 0] is black, [1 1 1] is white.

% A b&w indexed image
[imBarbara, cmapBarbara] = imread('barbara.bmp');
% imBarbara is the image, cmapBarabara is the colormap
figure('Position',[500 200 650 600]);
image(imBarbara); % plots the image (uses default colormap)
colormap(cmapBarbara); % sets the Barbara colormap
colormap('jet'); % matlab has some built in colormaps;: doc colormap
c = colormap; % returns current colormap

% A color indexed image
[imMarbles, cmapMarbles] = imread('marbles.gif');
image(imMarbles);
colormap(cmapMarbles);
% or 
colormap(cmapBarbara); 

% True RGB images are stored as 3 dimensional (m-by-n-by-3) arrays.
% A(:,:,1), A(:,:,2), and A(:,:,3) are intencities of Red, Green, and Blue.

% A true RGB image
imHorse = imread('horse.jpg');
image(imHorse); % rgb image (no need for a colormap)

% Once you've loaded an image, you can manipulate it as a matrix
% when you're done, you can write it back out using imwrite
% for indexed images:
% imwrite(x, colormap, filename)
% for true RGB color:
% imwrite(x, filename)

% e.g. 
imwrite(imBarbara, colormap('copper'), 'newBarbara.jpg');

%% Linear Algebra

% Basic function:
% rank(A)
% det(A)   % expensive
% inv(A)   % expensive
% norm(A)
% orth(A) % Orthonormal basis for range of matrix
% null(A) % Orthonormal basis for the null space of a matrix

%% Solving Linear Systems Ax=b

clc;clear;close all;

A = [1, 2; 3, 4]
b = [1; 2]
x=A\b   % "solves" Ax=b 
% Remark: x=b'/A  solves xA=b' 

%{
There are three cases:
- no solution: "\" finds a least squares solution.
- unique solution: "\" finds the exact solution. 
- infinetely many solutions: "\" finds a particular solution (with the
                             smallest Euclidean norm).
%} 

%{
General Solution:
The general solution to a system of linear equations Ax = b describes all 
possible solutions. You can find the general solution as follows:
(1) Check consitency: 
If rank(A)==rank([A,b]) => solution exists (1 or inf many)
If rank([A,b])>rank(A) => no solution => x=A\b finds a least squares "solution".
%}

% Example: overdetermined system, m>n (more equatoins than unknowns).
A=rand(11,10);  % 11 equations, 10 unknowns
b=rand(11,1);
rank(A)
rank([A,b])
x=A\b   % least squares solution
bar([A*x,b])  % A*x does not equal to b, but the two vectors are close.

% Suppose the system is consistent, i.e. rank(A)==rank([A,b])

%{
(2) Check uniqueness: 
If rank(A)== n, where [~,n] = size(A) => solution is unique and x=A\b.
If rank(A)<n => there are inf many solutions.
%}

% Example: unique soulution
A=rand(10,10);
b=rand(10,1);
rank(A)
rank([A,b])
[~,n] = size(A)
x=A\b
bar([A*x,b])
norm(A*x-b)  % length of the vector A*x-b

% Suppose the solution is not unique.

%{
(3) General solution of Ax=b is the sum of a particular solution x0 of Ax=b
plus a linear combination of the basis vectors v1,...,vk (k=n-r) for the 
solution space of the the corresponding homogeneous system Ax = 0. 
x0=A\b : gives a particular solution of Ax=b
null(A): returns a basis for the solution space to Ax = 0
%}

% Example: undetermined system, m<n (more unknowns than equations).

A=rand(5,10);  % 5 equations, 10 unknowns
b=rand(5,1);
rank([A,b])
rank(A)
[~,n] = size(A)
x0=A\b
norm(A*x0-b)   % check
V=null(A)

% another solution:
x=x0+V(:,1)+V(:,2)+V(:,3)+V(:,4)+V(:,5);
norm(A*x-b)

%% Square nonsingular matrices: inv(A)*b vs Gaussian Elimination vs A\b

close all;clear;clc;

% The naive way (bad, since computing the inverse matrix is time consuming)
naiveSolve = @(A,b) inv(A)*b;
[ts_naive,ns_naive] = generateTiming(naiveSolve); % measures performance
% ns_naive = sizes of test systems
% ts_naive = time (in sec) required for solving  

% Using Gaussian Elimination
% Gauss.m
[ts_Gauss,ns_Gauss] = generateTiming(@Gauss);

% The standard way: "\"
% The backslash operator attempts to decide what solution method
% will work well. It should be your first choice most of the time.
standardSolve = @(A,b) A\b;
[ts_std,ns_std] = generateTiming(standardSolve);

close all;
figure('Position',[500 200 650 600]);
loglog(ns_naive,ts_naive,'ro-',ns_Gauss,ts_Gauss,'m+-',ns_std,ts_std,'b+-');
xlabel('n');
ylabel('t [seconds]');
legend('Naive','Gauss','Standard');

% If you want more control and options, check "linsolve".

%% Condition Number 
% The condition number tells you a bit about how easy it is to solve a linear
% system on a computer. Suppose we want to solve the equation Ax = b, and our
% matrix A looks like this:
clc; clear; close all;
A = [1 4; 2 8]
% A is not invertible, so there is either no solution, or infinitely many
% solutions.
b = 1000*rand(2,1);    
rank(A)==rank([A,b]) % so there is no solution
x=A\b                         
% In this example, the least squares solution is not unique:
% any vector such that x1+4*x2=(b1+2*b2)/5 is a least squares solution

% But what if we change A just a bit?
A = [1 4; 2 8.0000000000001];  % nonsingular, theoretically solution exists. 
x=A\b
res= norm(A*x-b)  % the norm of the residual
% the residual is rather large, for even a 2 x 2 system
% This is reflected in the fact that the condition number for the matrix A
% is large. If A is singular, the condition number is (theoretically)
% infinite.
condNum=cond(A)  % large condition number is bad.

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
norm(A)         % if given no second argument, the default is "2"
norm(A,2)       % p-norm: sup norm(A*x,p)/norm(x,p)
norm(A,'fro')   % Frobenious norm = (sum |a_ij|^2)^(1/2)

%% Eigenvalues
close all;clear;clc;
% Matlab has a good eigenvalue solver called "eig".  This is different than
% "eigs" (which we will cover below; the "s" stands for "sparse", and
% not to pluralize "eig").  "eig" does different things, depending on the
% number of outputs: 
% - If we only have one output, it just returns the eigenvalues in a vector. 
% - If we have two outputs, it gives the eigenvalues (this time, as the 
%   diagonal of a matrix), as well as the eigenvectors.
clear; 
A = randn(3);
%The eigenvalues of A are:
lambda=eig(A)
% (recall that eigenvalues of a real matrix are allowed to be complex; if
% the matrix is symmetric, then we're guaranteed to get real
% eignevalues).

[V,D] = eig(A)
% The eigenvalues of A are on the diagonal of matrix D and the eigenvectors 
% are the columns of matrix V, so that A*V = V*D.

% Remark: If you don't need the eigenvectors, don't ask for them!  
% It will slow down the computation.

%% Eigenvalues for sparse matrices
clear; clc; close all;
% The command "eigs" finds only the few largest (or smallest) eigenvalues
% of a matrix.  It uses an entirely different kind of algorithm; this
% algorithm is very well-suited to sparse matrices.  Here, we'll
% demonstrate with a very large sparse matrix.
B = sprandsym(1000,.01);    % this is a 1000 x 1000 sparse symmetric matrix
                            % 1% of its entries are randomly selected to
                            % be nonzero; on those entries, the value is
                            % chosen from a uniform distribution.
spy(B); title('Sparsity pattern of the random symmetric matrix');
% Now, ask for the top and bottom 5 eigenvalues:
% (by "top", we mean those with the greatest magnitude)

disp('largest eigenvalues:')
D = eigs(B,5) 
% eigs(B) returns a vector of B's six largest magnitude eigenvalues
disp('smallest eigenvalues:')
D = eigs(B,5,'sm')  % "sm" stands for smalest magnitude

% Remark: you can use this command to also get the eigenvectors, e.g.,
% [V,D] = eigs(B,5).

%% Matrix Decompositions
close all;clear;clc;

A = rand(3);

% PLU decomposition
[L,U,P]=lu(A) % the permuted LU decomposition  P*A=L*U

% Cholesky Decompostions
U=chol(A) % produces an upper triangular matrix U, satisfying U'*U=A. 
          % chol assumes that A is symmetric. If it is not, chol uses the 
          % transpose of the upper triangle as the lower triangle. 
          % Matrix A must be positive definite: x'*A*x>0 for none zero x.
L = chol(A, 'lower') % L is lower triangular, L*L'=A.

% Orthogonal-triangular decomposition 
[Q,R] = qr(A)  %  produces an upper triangular matrix R and an 
               %  othogonal matrix Q so that A = Q*R.
               %  This decomposition can be also used for solving linear 
               %  systems: it is more computationally expensive than PLU
               %  decomposition, but more numerically stable. 

%% Next time, we will start with SVD, the most important matrix decomposition.

