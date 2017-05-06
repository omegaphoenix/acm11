% Simulate two random cuts of a stick of wood to calculate the
% probability that the three pieces form the three edges of a triangle.

% Number of samples
N = 10^5;

% Generate random cuts and then sort each pair of cuts
cuts = rand(2, N);
sortedCuts = sort(cuts);

% Calculate edge lengths
edgeLengths = [sortedCuts(1,:); 1 - sortedCuts(2,:); sortedCuts(2,:) - sortedCuts(1, :)];
% A triangle is formed if the sum of the two shorter edges > the longest
% edge which only occurs when the longest edge < 0.5
sortedLengths = max(edgeLengths) < 0.5;

% Display probility that the three pieces form the three edges of a
% triangle
prob = sum(sortedLengths) / N;
fprintf(['The empirical probability that the three pieces form the ' ...
  'three edges of a triangle is %f\n'], prob);
