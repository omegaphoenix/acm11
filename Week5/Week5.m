% ACM 11: Introduction to MATLAB   

%% Week 5: Eigenvalues, Matix Decompositions, SVD, Regression Analysis  
%%         Vectorization, Debugging, and Profiling.

%% Eigenvalues
close all; clear; clc;
% Matlab has a good eigenvalue solver called "eig". This is different than
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
% [V,D] = eigs(B,5)

%% Matrix Decompositions
close all;clear;clc;

A = rand(3)+eye(3);

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

%% Singular Value Decomposition (SVD) - the king of matrix decompositions
%{
The SVD is vaguely like a generalization of the eigenvalue decomposition; 
it gives slightly different information, and it is also more general 
(i.e. it applies to non-square matrices).  
We factor a matrix A as the product of three matrices: A = U*S*V'. 
If A is m-by-n, then 
- U is m-by-m, unitary U*U'=U'*U=I
- S is m-by-n, it is zero except on the diagonal. The nonzero diagonal 
  entries of S are called the singular values, and they are always real 
  and positive. By convention, they are always listed in decreasing order.
- V' is n-by-n, unitary V*V'=V'*V=I

In Matlab, the function to compute the SVD is called "svd()", and there is 
also a sparse version ("svds()").  
%}

clc; clear; 
A = randn(3,5);
% The vector of singular values of A are:
s = svd(A)
% All factors of the decomposition:
[U,S,V] = svd(A)
norm(A-U*S*V')  % the norm of the residual

% Example application of SVD to image compression:
% retain only the top singular values and vectors

close all; clear; clc;
load durer % Albrecht Durer's Melancholia; Matlab built-in image
% See: http://www.mathworks.com/help/matlab/learn_matlab/plotting-image-data.html
figure('Position',[500 200 650 600]);
image(X); 
colormap(map);
axis image
title('Original Image')

[U,S,V] = svd(X);

figure('Position',[500 200 650 600]);
spy(S) % view the sparsity pattern of the singular value matrix
title('Sparsity pattern')
bar(diag(S));  % singular values
ind = ceil(length(diag(S))*0.1); % take 10% of the singular values
S(ind:end, ind:end) = 0;
figure('Position',[500 200 650 600]);
spy(S)
title('New sparsity pattern')
figure('Position',[500 200 650 600]);
Y=U*S*V';
image(Y)
colormap(map)
axis image
title('Compressed Image')

%% Simple Linear Regression
% Explain the method in class.
close all; clear; clc;
% Let us genenerate some data with linear trend:
x=(0:.01:2).';
y = 6*x + 2 + 0.5*randn(size(x));
figure('Position',[500 200 650 600])
plot(x,y,'ro');

% One way to fit a line to the data is to use the basic fitting GUI
% Tools >> Basic fitting

% Another way is to use polyfit: polynomial curve fitting
% p = polyfit(x,y,n) returns the coefficients for a polynomial p(x) of 
% degree n that is a best fit (in a least-squares sense) for the data (x,y). 
p = polyfit(x,y,1);

yfit = polyval(p,x); % polyval(p,x) returns the value of a polynomial p 
                     % evaluated at x.
hold on;
plot(x,yfit,'-b');

%% Multiple Linear Regression
% Explain the method in class.
% B = regress(y,X) 
% - X is an n-by-p matrix of p predictors at each of n observations.
% - y is an n-by-1 vector of observed responses.
% - B is a p-by-1 vector B of coefficient estimates for a multiple linear 
%     regression of the responses in y on the predictors in x.  

close all;clear;clc;
load carbig % a dataset that contains various measured variables for 406 
            % automobiles from the 1970's and 1980's.
x1 = Weight;
x2 = Horsepower;
y = MPG; %  Miles per gallon

%Compute regression coefficients for a linear model with an interaction term:
X = [ones(size(x1)), x1, x2, x1.*x2];
b = regress(y,X);

%Plot the data and the model:
figure
scatter3(x1,x2,y)
xlabel('Weight')
ylabel('Horsepower')
zlabel('MPG')
hold on
x1fit = min(x1):100:max(x1);
x2fit = min(x2):10:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1) + b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;
mesh(X1FIT,X2FIT,YFIT)

% More tools for data analysis: in Statistics and Machine Learning Toolbox

%% Vectorization

clear; clc; close all; 

%{
MATLAB is optimized for operations involving matrices and vectors.  
"Vectorization" is the process of revising loop-based, scalar-oriented 
code to use MATLAB matrix and vector operations.
Vectorizing your code is worthwhile for several reasons:
(1) Performance: Vectorized code often runs much faster than the corresponding 
code containing loops.
(2) Less Error Prone: Without loops, vectorized code is often shorter. 
Fewer lines of code mean fewer opportunities to introduce programming errors.
(3) Appearance: Vectorized mathematical code appears more like the 
mathematical expressions found in textbooks, making the code easier to understand.

"Vectorize!" is one of the mantras of MATLAB programming.
%}

% Trivial example: here is a non-vectoriaed code
i = 0;
for t = 0:.01:10
    i = i + 1;
    y(i) = sin(t);
end
% Here is a vectorized version of the same code:
t = 0:.01:10;
y = sin(t);

% Another example: generate an n-by-n matrix B of polynomial basis functions
n = 5000;
x = linspace(0,1,n);
% Bad
tic
B = NaN(n,n); % preallocation
for i=1:n
    for j=1:n
        B(i,j) = x(j)^(i-1);
    end
end
toc

% Better
tic
A=repmat(x,n,1);
C=repmat((0:n-1)',1,n);
B = A.^C;
toc

% Strategy: matrix thinking + built-in functions

%% Helpful Built-In Functions with Examples

clear; clc;
% all(v)  Tests if all elements of v are nonzero (or true)
% any(v)  Tests if at least one elements of v are nonzero (or true)
% Illustration: 
a=1;
b=2;
x=a+(b-a)*rand(10,1); 
y=a+(b-a)*randn(10,1);
all(x>=a)
all(y>=a)
any(x<a)
any(y<a)

% Application: return only the rows of A that don't have negative elements
A=[1, 2, -3; 4, 5, 6; 7, 8, -9]
B = A(~any(A<0,2),:)  % work it out on a paper and to see what is going on!
                      % what is A<0 ?
                      % what is any(A<0) ?
                      % what is any(A<0,1) ?
                      % what is any(A<0,2) ?
                      % what is ~any(A<0,2) ?
%----------------------------------------------------------
% cumsum(a): finds cumulative sum of elements of a
% Illustration:
clear; clc;
A = 1:5
B = cumsum(A)

% Application: Simulation of Brownian motion
% W(k,dt)=sqrt(dt)*(Z1+Z2+...+Zk), k=1,...
% dt - time step, Zj - independent N(0,1) variables
T=1; N=10000; dt=T/N; 
tic
%  W(k,dt)=W(k-1,dt)+ sqrt(dt)*Zk;
dW=zeros(1,N);
W=zeros(1,N);
dW(1) = sqrt(dt)*randn;
W(1) = dW(1);
for j = 2:N
    dW(j) = sqrt(dt)*randn;
    W(j) = W(j-1) + dW(j);
end
toc
figure
plot([0:dt:T],[0,W])

tic
dW = sqrt(dt)*randn(1,N);
W = cumsum(dW);
toc
hold on
plot([0:dt:T],[0,W])

%------------------------------------------------------------
% diff(x):  calculates differences between adjacent elements of x
% b=diff(a), then b(i)=a(i+1)-a(i).
% Illustration:
clear; clc; close all;
A = [1, 2, -2, 1]
B = diff(A)
% find(x) returns a vector of indices of nonzero elements in x
A=[0 0 1 1 0 1]
B=find(A)

% Application: find the local maxima in an array
a = randn(100,1);
b = diff(a); %size 99 by 1
ind = find(b(1:end-1)>0 & b(2:end)<0);
figure
plot(a);
hold on;
plot(1+ind,a(1+ind),'rx')

% How to find local minima? 
% ind = find(b(1:end-1)<0 & b(2:end)>0);

%--------------------------------------------------------------
% prod(A) returns the product of the array elements of A.
% Illustration:
a=[1,-1,1,-1];
b=prod(a)

% Application: factorial function
clear; clc; 
N=10000;
n=100;
tic
for i=1:N
    factorial(n);
end
toc
tic
for i=1:N 
    prod(1:n);
end
toc

% Even though factorial is built-in, it uses cumprod

%----------------------------------------------------------------
% B = reshape(A,size) reshapes A using the size vector, size, to define size(B). 
% For example, reshape(A,[2,3]) reshapes A into a 2-by-3 matrix. 
% size must contain at least 2 elements, and prod(sz) must be the same as numel(A)
A = rand(4)
B = reshape(A,2,[]) % B has 2 rows and the automatically calculated # columns
% If A=rand(5), then B = reshape(A,2,[]) leads to an error.

%-----------------------------------------------------------
% B = sort(A) sorts the elements of A in ascending order 
% - If A is a vector, then sort(A) sorts the vector elements.
% - If A is a matrix, then sort(A) treats the columns of A as vectors 
%   and sorts each column.
% - sort(A,dim) returns the sorted elements of A along dimension dim.
A=[1, -1, 0; -2, 3, -5]
sort(A)
sort(A,1)  % sorts columns
sort(A,2)  % sorts rows

%% Debugging
% Debugging is the process of locating and fixing "bugs" (errors).
% In Matlab it is most useful in conjunction with "breakpoints."  
% To add a breakpoint into a line of code, just click the horizontal line 
% to the left of the line of code. It should turn into a red dot. The dot
% turns gray if the code has been changed and not saved.  When you run the
% code, the execution pauses at the breakpoint everytime it reaches the
% line.  The Matlab command menu now has a different look to it: "K>>".
% You can see the local variables to the function, and hopefully
% piece-together what went wrong!  To step through the code, or let it run,
% see the various options (and the shortcuts) in the "DEBUG" menu of the
% EDITOR.  You can also control these with commands, e.g. to quit the
% debugger, use "dbquit".

%% Profiling: finding the bottlenecks in your code
clear; clc;
% MATLAB has a very convenient tool to find bottlenecks and inefficiencies
% in you programs
profile on
sample_script
profile off
profile viewer
