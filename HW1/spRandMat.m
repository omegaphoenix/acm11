function randSymPosSparseMat = spRandMat(n)
% SPRANDMAT Produce a random symmetric, positive definite, sparse n x n matrix

% Construct a random matrix A with non-zero density of 0.05
density = 0.05;
randomMat = sprand(n, n, density);

% Return A^T A + I
randSymPosSparseMat = transpose(randomMat) * randomMat + eye(n);

end
