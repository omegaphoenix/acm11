function L = discreteLaplace(ny, nx, hy, hx)
% DISCRETELAPLACE Return approximation of Laplace operator

% Side of matrix corresponding to Laplace operator
nxny = nx * ny;

% Create ones matrix to build shift operators
onesMat = ones(nxny, 1);

% First and third columns are upper and lower shift operators when placed in
% the diagonals just off of the main diagonal.
xApprox = [onesMat -2*onesMat onesMat];
xApprox = xApprox / hx / hx;
% Left and right shift operators when on the ny-th diagonal from the main
% diagonal
yApprox = [onesMat -2*onesMat onesMat];
yApprox = yApprox / hy / hy;

% Combine x and y approximations
B = [yApprox(:,1) xApprox(:,1) (xApprox(:,2) + yApprox(:,2))...
      xApprox(:,3) yApprox(:,3)];
% How far to shift diagonals
diag = [-ny -1, 0, 1 ny];
L = spdiags(B, diag, nxny, nxny);
end
