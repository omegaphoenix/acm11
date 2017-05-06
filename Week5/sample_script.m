% Sample script
clc;clear;close all;
n = 1000;
A = randn(n);
for p = 1:40
    matrix_mult_example(A);
end
maxI=200;
A = randi(maxI,n); % i stands to integer
for p = 1:40
    matrix_mult_example(A);
end
