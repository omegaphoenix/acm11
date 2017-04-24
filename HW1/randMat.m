function randSymPosDenseMat = randMat(n)
% RANDMAT Produce a random symmetric, positive definite, dense n x n matrix

% Construct a random matrix A uniformly distributed on the interval [0, 1]
randomMat = rand(n);
% Change the interval to [-1, 1]
randomMat = (randomMat * 2) - 1;

% Return A^T A + I
randSymPosDenseMat = transpose(randomMat) * randomMat + eye(n);

end
