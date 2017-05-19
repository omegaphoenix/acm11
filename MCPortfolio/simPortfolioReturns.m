function [firstDecile, mid, lastDecile] = simPortfolioReturns(portSize, ratio)
% SIMPORTFOLIORETURNS Return 10, 50, and 90 percentile of portfolio returns
% calculated using a Monte Carlo simulation with 1000 trials of 80 years
% portSize is the original portfolio size
% ratio is a 4-vector with the relative ratios of the 4 funds in the portfolio
% The four funds are VTI (total US stock market), VXUS (total international
% stock market), BND (total US bond market), and BNDX (total international
% bond market).
% e.g. simPortfolioReturns([1, 1, 1, 1]) is the equal weighted portfolio
% Assume that {TICKER}_PRICE_2013_2015.txt files exist and have the same
% date range. We could only go back to 2013 because that was the inception
% date of the international bond fund, BNDX.

% Count funds
tickers = {'VTI', 'VXUS', 'BND', 'BNDX'};
numFunds = numel(ratio);
assert(numel(ratio) == numel(tickers));

% Set random seed for Monte Carlo
rng(numFunds);

% Import adjusted price data
for i=1:1:numFunds
  filename = strcat(tickers{i}, '_PRICE_2013_2015.txt');
  temp = importdata(filename, ' ', 15);
  prices(:, i) = [temp.data(:, [end])];
end

% Convert prices to returns
absReturns = diff(prices) ./ prices(1:end-1, :);
logReturns = diff(log(prices));

% Build covariance matrix
absCov = cov(absReturns);
M = cov(logReturns);

% Calculate expected returns using capital asset pricing model (CAPM)
% which assumes more risk/variance leads to greater returns
% betas are with respect to the US stock market (VTI)
riskPremium = .04; % assume stocks return 4% more risk free rate
riskFreeRate = .0265; % Return on 20 year T-bonds

% beta = cov(A, M) / var(M) where M is the US stock market
betas = absCov(:,1) / absCov(1,1);
expectedReturns = betas * riskPremium + riskFreeRate;
expLogReturns = log((1 + expectedReturns).^(1/365));

% Check if positive definite before taking Cholensky factor
positiveDefinite = all(eig(M) > 0)
if (~positiveDefinite)
  epsilon = 1e-20
  M = M + (epsilon * eye(numFunds))
end
% Calculate Cholensky factor
L = chol(M, 'lower');

% Generate numFunds-vectors for every day of the simulation
marketDaysPerYear = 250;
years = 30;
numTrials = 1000;
% Generate normally distributed random numbers
s = randn(numFunds, marketDaysPerYear * years * numTrials);
% Generate return-vectors
r = L*s; % add variance
for i = 1:numFunds
  r(i,:) = r(i,:) + expLogReturns(i);
end

% Normalize weight vector
w = ratio / sum(ratio);
% Transpose if necessary to handle input in either row or column vector form
if length(w) ~= numFunds
  w = w';
end
% Dot product of weight vector and return-vectors
R = w*r;

% Add 80 years of log returns
R = reshape(R, [years * marketDaysPerYear, numTrials]);
R = sum(R);

% Print some useful daily log return statistics
minimum = min(R);
maximum = max(R);
mu = mean(R);
med = median(R);
stdev = std(R);
skew = skewness(R);
Cov = cov(r');
excessKurtosis = kurtosis(R);
% Jarque-Bera statistic - jbstat
[h, p, jbstat, critval] = jbtest(R);
% Chi-Squared probability - p
[h, p, stats] = chi2gof(R);
% off diagonals are serial correlation
serialCorr = (corrcoef(R(2:end), R(1:end-1)));
% 99% VaR
ninetyNineVar = prctile(R, 1);
% 99% Expected Shortfall
shortfall = mean(R(R <= ninetyNineVar));

% Calculate final portfolio values at 10, 50, 90 percentiles
firstDecile = exp(prctile(R, 10)) * portSize
mid = exp(median(R)) * portSize
lastDecile = exp(prctile(R, 90)) * portSize

end
