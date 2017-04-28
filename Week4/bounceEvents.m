function [value,isterminal,direction] = bounceEvents(t,Y)
% Locate the time when height passes through zero in a decreasing direction
% and stop integration.  
value = Y(2);     % detect height = 0 
isterminal = 1;   % stop the integration
direction = -1;   % directions