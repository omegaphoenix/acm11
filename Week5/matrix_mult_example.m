function C = matrix_mult_example(A)
% Function MATRIX_MULT_EXAMPLE multiplies a given matrix A of size n-by-n
% by a randomly generated diagonal matrix D and then matrix B of the same size
% and returns the result as matrix C
n=size(A,1);

B = randn(n);
D = diag(randn(n,1));
   
%C = A*D*B;
C = A*(repmat(diag(D),1,n).*B);

end

