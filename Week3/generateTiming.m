function [times, nValues] = generateTiming(g)
% generateTiming runs a (fixed) set of tests for a function g
%   g must be a function g(A,b) where A is a square matrix and b is a
%   compatible column vector

    nValues = floor(logspace(2,4,20)); % matrix sizes
    % logspace(a,b,n) generates n logarithmically spaced points between 10^a and 10^b.
    nTests = length(nValues); % number of tests 
    times = NaN(nTests,1);    % preallocation
    
    for i=1:nTests
       % Generate Test System:
       n = nValues(i);
       A = randn(n);
       b = randn(n,1);
       % How long the solutions takes
       tic;
       g(A,b);
       times(i) = toc;
    end

end