function finalPortfolioSizes =...
  simPortfolioReturns(portSize, ratio, percentiles, years)
% SIMPORTFOLIORETURNS Return percentiles of portfolio returns
% calculated using a Monte Carlo simulation with 1000 trials
% - portSize - the original portfolio size
% - ratio - a 4-vector with the relative ratios of the funds in the portfolio
% - percentiles - a vector of percentiles to return
% - years - number of years to simulate
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
    temp = importdata(filename, ' ', 15); % first 15 lines are not data
    prices(:, i) = [temp.data(:, [end])]; % assume dates are the same
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
positiveDefinite = all(eig(M) > 0);
% not positive definite if two stocks were perfectly correlated
if (~positiveDefinite)
    fprintf(['Covariance matrix is not positive definite. Did you include'...
            'the same stock twice?  \n']);
    % small value to make matrix positive definite without affecting results
    epsilon = 1e-20;
    M = M + (epsilon * eye(numFunds));
end

% Calculate Cholensky factor
L = chol(M, 'lower');

% Generate numFunds-vectors for every day of the simulation
marketDaysPerYear = 250;
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

% Add *years* years of log returns
R = reshape(R, [years * marketDaysPerYear, numTrials]);
R = sum(R);

    function finalVal = calcPortfolio(logReturn)
    % CALCPORTFOLIO Return total portfolio size.
    % logReturn - the total log return
        finalVal = exp(logReturn) * portSize;
    end

% Print some useful statistics
minimum = calcPortfolio(min(R));
fprintf('After %d years with an initial investment of $%.2f:\n',...
        years, portSize);
fprintf('Minimimum portfolio size was $%.2f\n', minimum);
maximum = calcPortfolio(max(R));
fprintf('Maximum portfolio size was $%.2f\n', maximum);
mu = calcPortfolio(mean(R));
fprintf('Average portfolio size was $%.2f\n', mu);
med = calcPortfolio(median(R));
fprintf('Median portfolio size was $%.2f\n', mu);

ninetyNineVar = prctile(R, 1);
ninetyNineVal = calcPortfolio(ninetyNineVar); % 99% Value at Risk
fprintf('99 percent of the time we will have more than $%.2f\n',...
        ninetyNineVal);

shortfall = mean(R(R <= ninetyNineVar));
shortfallVal = calcPortfolio(shortfall); % 99% Expected Shortfall
fprintf('When we have less than that, we have on average $%.2f\n',...
        shortfallVal);

% Calculate final portfolio values at specified percentiles
finalPortfolioSizes = {};
for i = 1:numel(percentiles)
  percLogReturn = prctile(R, percentiles(i));
  finalPortfolioSizes = [finalPortfolioSizes, calcPortfolio(percLogReturn)];
end

end
