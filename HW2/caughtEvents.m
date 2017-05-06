function [value, isterminal, direction] = caughtEvents(t,Y)
% Locate the time when the raptor eats you since he is within 0.1 m
% and stop integration.
value = norm(Y) - 0.1 % detect within distance of 0.1 m
isterminal = 1; % stop the integration
direction = 0; % you are eaten and the raptor is eating
