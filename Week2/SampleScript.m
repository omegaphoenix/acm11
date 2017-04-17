% A script which generates a sparse Vandermonde matrix

close all; clear; clc;

n = 10^2;  % matrix size

fprintf('Generating the Vandermonde matrix for a random vector of size %d-by-%d\n', n, n);

tic

x=rand(n,1); % random vector

% Bad idea:
%{
for i=1:n
    for j=1:n
        A(i,j)=x(i)^(j-1);
    end
end
%}
A=fliplr(vander(x));   % vander(x): an alternate form of the Vandermonde 
                       % fliplr: flips array left to right
                       
A = A.*(A>0.01); 

fprintf('Generation took %f seconds \n', toc);

pause(3);  % pauses execution for 3 seconds before continuing

disp('Visualizing Vandermonde matrix.');

pause(3); 

tic

spy(A); % note the number of NONzeros - can also be called with nnz(A)

fprintf('Visualization took  %f seconds \n', toc)