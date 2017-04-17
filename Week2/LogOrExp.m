function y = LogOrExp(x)
% LogOrExp returns the value of log(x) if x<1 and exp(x) if x>=1

if x>=1
    y = exp(x);
else
    y = log(x);
end


