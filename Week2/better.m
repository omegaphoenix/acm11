% This script computes a solution to a linear system Ax=b, where 
% - A is tridiagonal with 4 on the diagonal, 1 on the sub- and super-diag., 
% - b is a random vector.
% It also computes the maximum magnitude entry in the residual Ax-b.

close all; clear; clc;

% System size
n = 1000;

% Generate a tridiagonal matrix
c = ones(n,1);
A = spdiags([c,4*c,c],-1:1,n,n); % creates an n-by-n sparse matrix by  
                                 % taking the columns of [c,4*c,c] and 
                                 % placing them along the diagonals 
                                 % specified by -1:1

% Generate a random rhs
b = rand(n,1);

% Solve 
x = A\b;

% Compute a residual norm
resInfNorm = norm(A*x-b,'inf');

% print
fprintf('For n=%d, the maximum magnitude entry in the residual is %e.\n',...
    n,resInfNorm);
