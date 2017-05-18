function [firstDecile, mid, lastDecile] = simPortfolioReturns(percents)
% SIMPORTFOLIORETURNS Return 10, 50, and 90 percentile of portfolio returns
% calculated using a Monte Carlo simulation

% Build covariance matrix
tickers = {'VTI', 'VXUS', 'BND', 'BNDX'};
for i=1:1:4
  filename = strcat(tickers{i}, '_PRICE_2013_2015.txt');
  temp = importdata(filename, ' ', 15);
  prices(:, i) = [temp.data(:, [end])];
end

logReturns = diff(log(prices));

M = cov(logReturns)

% Check if positive definite before taking Cholensky factor
positiveDefinite = all(eig(M) > 0)
if (~positiveDefinite)
  epsilon = 0.0001
  M = M + (epsilon * eye(4))
end
L = chol(M, 'lower');

% Set random seed
rng(4);
% Generate 4-vectors for every day of the simulation
marketDaysPerYear = 250;
years = 80;
numTrials = 1000;
s = randn(4, marketDaysPerYear * years * numTrials);
% Generate return-vectors
r = L*s;

% Dot product of weight vector and return-vectors
w = percents' / sum(percents);
R = w*r;

% Statistics in percentage
size(R)
minimum = min(R)
maximum = max(R)
mu = mean(R)
med = median(R)
stdev = std(R)
skew = skewness(R)
Cov = cov(r')
excessKurtosis = kurtosis(R)
% Jarque-Bera statistic - jbstat
[h,p,jbstat,critval] = jbtest(R)
% Chi-Squared probability - p
[h,p,stats] = chi2gof(R)
% off diagonals are serial correlation
serialCorr = (corrcoef(R(2:end),R(1:end-1)))
% 99% VaR
ninetyNineVar = prctile(R,1)
% 99% Expected Shortfall
shortfall = mean(R(R <= ninetyNineVar))
firstDecile = 0
mid = 0
lastDecile = 0

end
