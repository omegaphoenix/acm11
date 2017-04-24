% Calculate time to solve different size linear systems constructed using
% dense random matrices.
% Sizes are n = 10, 100, 1000, 2000.

% Clear and close figures
clear; close all; clc;
% Size of matrices to solve
nValues = [10, 100, 1000, 2000];
for n = nValues
    S = randMat(n);
    b = rand(n, 1);
    % Solve linear system and time it
    tic;
    x = S\b;
    timeElapsed = toc;
    % Print time it tookk to solve the linear system
    fprintf('Took  %f time to solve linear system with n=%i\n', ...
        timeElapsed, n);
end
