function [y,z] = LogOrExp2(x,b)
% LogOrExp2(x) returns the value of log(x) if x<1 and exp(x) if x>=1. 
% With a second input variable, LogOrExp2(x,b) returns the
% logarithm of x base b if x<1 and exp(x) if x>=1. 
% If second output variable is requersted, [y,z] = LogOrExp2(x,b),
% then the function also returns the value z = 2^x.

if nargin==1  % nargin =  Number of function input arguments
    logarithm_base = exp(1); % this is the number e
end
if nargin==2  % nargin =  Number of function input arguments
    logarithm_base = b;
end
    
if x>=1 
    y = exp(x);
else
    y = log(x)/log(logarithm_base); % the log of x base b
end

if nargout==2 % nargout = Number of function output arguments
    z = 2^x;   
end