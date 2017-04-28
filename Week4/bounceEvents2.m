function [value,isterminal,direction] = bounceEvents2(t,Y)
% Locate the time when height passes through zero in a decreasing direction
% or x outside of the unit interval and stop integration.  
value = [Y(2), Y(1), 1-Y(1)];    % detect height = 0 or x outside of the unit interval
isterminal = [1 1 1];            % stop the integration
direction = [-1 -1 -1];          % directions