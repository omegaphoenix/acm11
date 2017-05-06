% Load dice data and determine which die is fair based on relative
% frequencies.
load('DiceData.mat');
faces = 1:6;

% Calculate relative frequencies of die 1
countDie1 = histc(die1, faces);
relFreqDie1 = countDie1 / numel(die1)

% Calculate relative frequencies of die 2
countDie2 = histc(die2, faces);
relFreqDie2 = countDie2 / numel(die2)

%% Results
%relFreqDie1 =
%    0.1664
%    0.1707
%    0.1611
%    0.1706
%    0.1675
%    0.1637

%relFreqDie2 =
%    0.0179
%    0.1545
%    0.3247
%    0.3279
%    0.1571
%    0.0179


% Die one is fair because all the faces occur with +/- 0.01 chance
% whereas for die two, the relative frequencies differ by over 0.3.
% We confirm this result below using a chi-squared test.

%% Chi Squared Test
expCounts = ones(6, 1) * numel(die1)/6;
[h1, ~, ~] = chi2gof(faces, 'Ctrs', faces, 'Frequency', countDie1, ...
  'Expected', expCounts)
[h2, ~, ~] = chi2gof(faces, 'Ctrs', faces, 'Frequency', countDie2, ...
  'Expected', expCounts)

% Print die 1 chi squared test results
fprintf('Using a Chi Squred Goodness of Fit test, we');
if (~h1)
  fprintf(' fail to');
end
fprintf([' reject the null hypothesis that die 1 is fair on a 5 percent' ...
  ' significance level.\n']);

% Print die 2 chi squared test results
fprintf('Using a Chi Squred Goodness of Fit test, we');
if (~h2)
  fprintf(' fail to');
end
fprintf([' reject the null hypothesis that die 2 is fair on a 5 percent' ...
  ' significance level.\n']);
