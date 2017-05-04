load('DiceData.mat');
faces = [1, 2, 3, 4, 5, 6];

% Calculate relative frequencies of die 1
countDie1 = histc(die1, faces);
relFreqDie1 = countDie1 / numel(die1)

% Calculate relative frequencies of die 2
countDie2 = histc(die2, faces);
relFreqDie1 = countDie2 / numel(die2)

% Die one is fair because all the faces occur with +/- 0.01 chance whereas
% for die two, the relative frequencies differ by over 0.3.