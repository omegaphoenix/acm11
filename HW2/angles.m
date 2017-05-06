% Find the best and worst angles to escape raptors

% Since the problem is symmetric on the axis of the third (injured)
% raptor which is at (0, 2a), we can solve for the minimum and maximum
% between -90 and 90 or 90 and 270. We take the modulus to get values
% between 0 and 360
f1 = @(x) timeToMeal(x); % function to find minimum
f2 = @(x) -1 * timeToMeal(x); % function to find maximum

%% Solve min and max in upper two quadrants
% Find min of timeToMeal on [-90 to 90]
[alphaMinUpper, ~] = fminbnd(f1, -90, 90);
% Find min of -timeToMeal on [-90 to 90]
[alphaMaxUpper, ~] = fminbnd(f2, -90, 90);

% Convert range to [270, 360] or [0, 90]
alphaMinUpper = mod(alphaMin1, 360);
alphaMaxUpper = mod(alphaMax1, 360);

%% Solve min and max in lower two quadrants
% find min of timeToMeal on [90, 270]
[alphaMinLower, ~] = fminbnd(f1, 90, 270);
% find min of -timeToMeal on [90, 270]
[alphaMaxLower, ~] = fminbnd(f2, 90, 270);

fprintf(['The angles to maximize time before you are caught are ' ...
  '%f and %f degrees.\n'], alphaMaxUpper, alphaMaxLower);
fprintf(['The angles which will get you devoured fastest are ' ...
  '%f and %f degrees.\n'], alphaMinUpper, alphaMinLower);
