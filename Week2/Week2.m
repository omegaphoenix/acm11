% ACM 11: Introduction to MATLAB

%% Week 2: Cells, Errors, Loops, Saving and Loading, Control, M-Files, Scripts, Functions, Figures, 2D-plotting
clc; clear; close all;

%% Cells

% Cell is also a Matlab datatype. A cell is like an array, except that
% every element of an array must be the same datatype (e.g. a character, or
% a double). Cells are not restricted like this. In syntax, the
% difference is that we use { and }; these curly braces are used for both
% creating and access elements of a cell.  For example,
myCell = {'First entry is a string', -.364, [1 2; 3 4], ones(5)}

% This is a useful way to visualize the contents of a cell"
cellplot(myCell);
title('Cell Visualization');

% For much more information on cells: doc Cell Arrays

%% Error Messages
%{
There are two things we'd like to do with errors:
(1) when we write our own code, we might want to "throw" an error, and
(2) when Matlab generates an error, we might want to do something special.
%}

% (1)  When we throw an error, execution will stop.
% The simplest way to throw an error is just using "error('message')"

error('Something Happened!')

% (2) Matlab produces useful error messages, which include line numbers
% when running a script or function

[1 2] + [1 2 3]  % Dimensions must agree!

% We can handle Matlab's errors using "try" and "catch".  
% Here's how:
try
    [1 2] + [1 2 3];  % because the dimension disagree  
                      % Matlab will generate an error.
catch
    % If an error is thrown, then the code here will be executed:
    disp('dimensions must agree :(')
end

%% Saving and Loading 
clear;
a=1; b=2; c=3;
save('examplestorage.mat', 'a', 'b') % store just the variables a and b
clear % clear the memory
load('examplestorage.mat') % loads back a and b
anothervar = 'Hello World!' % make a new variable
save examplestorage % alternative format for saving and loading entire workspace
clear
load examplestorage

%% Loops and Control
% These are intuitive but expensive. 
% This is an example of a for loop:
clear; 
for i=1:10
    a(i)=i^2;
end

% you can do this, but it's hard to read
for i=1:10 a(i)=i^2; end

% Remark: this would not work (we need a semicolon after a(i)=i^2):
% for i = 1:10 a(i)=i^2 end

% But this would work
for i=1:10
    a(i)=i^2
end

% General rule: try to avoid loops (they are time consuming)
% Example: Suppose we want to define a matrix A(i,j)=i*j.
% Here is the most straightforward way:
N=1000; 
tic
for i=1:N
    for j=1:N
        A(i,j) = i*j;
    end
end
toc

% But we can avoid loops:
clear A;
tic
A = [1:N]'*[1:N];
toc

% The "while" statement is very similar to a "for" statement.  
% The idea is that the code is executed UNTIL some condition becomes false. 
% The syntax is:
%
% while [test]
%   [body of code]
% end
%
% and it will execute until the test is false.

a = 1;
while a < 100
   disp(a);
   a = a*2; % MATLAB doesnt't have *= and similar operators
end

% "If" statement:
if 1 > 0
    disp('1 is greater than 0');
end

% "else"
% Syntax:
% if [test]
%   [body of code]
% else
%   [body of code]
% end

% Example: This piece of code emulates coin flipping 
clc
for i=1:10
    if rand > 0.5
        disp('heads');
    else
        disp('tails');
    end 
end

% To do equality comparisons, make sure to use TWO "=" signs:
x = 5;
if x == 10
    disp('x is 10');
else
    disp('x is not 10');
end

% ~= is "not equal"
if x ~= 10
    disp('x is not 10');
else
    disp('x is 10');
end

% All loops and controls are slow.
% Example: 
% We have to vectors x and y and we want to construct a new vector z with
% z(i)=2*x(i) if x(i)>y(i), and 
% z(i)= -y(i) if x(i)<= y(i). 
clc; clear;
N = 10^7;
y = randn(1,N);
x = randn(1,N);
% The plain way to create z:
tic
for i = 1:N
    if x(i) > y(i)
        z(i) = 2*x(i);
    else
        z(i) = -y(i);
    end
end
time1=toc
% the "MATLAB-way" to create z:
tic
z = 2*x.*(x>y) - y.*(x<=y);
time2=toc
% faster!

% switch  ("generalization" of if/else)
a = 'ACM 11 is fun';
switch a %<--- a scalar or a string
    case 1,
        disp('1');
    case 2,
        disp('2');
    case 'ACM 11 is fun',
        disp(a);
    otherwise,
        disp('ACM 11 is not fun');
end

% break: terminates execution of "for" or "while" loop
a = 0
while 1
    a = a + 7;
    disp(a);
    if mod(a,5)==0   % mod(a,b) gives the remainder after division of a by b
       break; % breaks the loop
    end
end

% continue: passes control to next iteration of "for" or "while" loop
clc
for i=1:10
   if i<10
       continue; %break; 
   end
   disp(i);
end

%% Preallocation: saves time

clc; clear;
% Suppose we want to create a vector a of size 10^7 with a(i)=2i-1.
% The most straightforward way:
n=10^7;
tic
for i=1:n
   a(i) = 2*i-1;  
end
toc

% Let's now preallocate some space for a:
clear a
tic
%a = zeros(1,n);   % common
a = NaN(1,n);    % slightly more efficient 
for i=1:n
   a(i) = 2*i-1; 
end
toc

% But it is of course much better to simly avoid loops (if possible)
clear a
tic
a = 1:2:2*n-1;  % range notation 
toc

%% M-files
% MATLAB uses m-files to define 
% - scripts (sequence of MATLAB statements in a file) and 
% - functions (have inputs and outputs).

% Scripts are simply files which list a series of commands to execute in
% order
% 
% see SampleScript.m

%% Functions

%{
Functions are different than scripts.  They take a specified input, 
and give an ouput.

A function is defined in a separate file (unlike the anonymous
functions that we  will discuss in the next cell), and this file should 
have the same name as the function. The syntax is:

function output = functionName(input)
[code...]

or, for several inputs/outputs:

function [output1, output2] = functionName(inputA, inputB)
[code...]

In the rest of the function body, all the variables that weren't passed
as inputs are "local". This means they are completely different variables
than variables of the same name elsewhere.  

There is no need to "end" the function, but you can if you wish; to do this,
just type "end" at the very end. 

Functions can have "sub-functions". These are available ONLY to the main
function, and not to any other part of MATLAB. To make a sub-function,
below the rest of the function code just use the same function syntax.  You
can make as many subfunctions as you want. Again, there is no need to "end"
a subfunction (but I would recommend doing this for clarity), because 
Matlab can determine this automatically; but, if you do, then you must 
do it for all subfunctions.

%}

% Examples of functions:

% see LogOrExp.m 

% The number of input/output variables can be different for a given function - 
% see LogOrExp2.m for manipulations with numbers of arguments. 

% Remark: help for the function is the first set of comments in the m-file.
help LogOrExp
help LogOrExp2
% or even
doc LogOrExp2

% Path
% When you type the name of a script/function, Matlab needs to know where to
% search for it. It can't search the entire hard drive, so it restricts
% itself to directories in the "path".  To see which directories are in your
% path, just type "path", although you're likely to get more output than
% you want.  There are many functions relating to the path.  The most
% useful are "addpath", which adds a certain path to the directory,
% and "pathtool", which is a graphical tool that lets you manipulate 
% the path.  The current directory is always in the path.  

%% Functions with Subfunctions
% Subfunctions can come in two flavors:
% ======= Flavor 1: ============
%{
function mainFunction
[body of mainFunction]  

function subFunction
[body of subFunction]
%}
% =======  Flavor 1, variation (same effect, different syntax) =====
%{
function mainFunction
[body of mainFunction]  
end

function subFunction
[body of subFunction]
end
%}

% In flavor 1, the subFunction is visible only to the mainFunction and 
% it can't see any of the local variables in mainFunction.

% ======= Flavor 2: ============
%{
function mainFunction
[body of mainFunction]

function subFunction
[body of subFunction]
end   <--- refefers to mainFunction, not subFunction
%}

% In flavor 2, the subFunction also visible only to the mainFunctin, but  
% but it can "see" all the local variables of mainFunction.  

%% Anonymous functions
% The @ symbol allows you to refer to a function.
% For example, @ can be used to refer to a built-in function:
@sin

% Or it can be use for defining an anonymous function
%  The syntax is 
% functioName = @(arg1,arg2,...) [expression inolvong arg1,arg2,...];
f = @(x,y,z) x+y+z;

% The anonymous functions are less flexible than functions 
% as defined in m-files so should be reserved for small tasks, e.g.,
q = integral(@(x) x.^2,0,1)
% Another example: q = integral(@sin,0,pi)
%% General Strategy: Use scripts to develop, then convert to functions

% In scripts, start with:
close all; clear; clc;

%% Style
% Writing clear and maintanable code - some tips:
%   1. Use functions liberally
%        a. function m-files
%        b. anonymous functions
%   2. Take the time to clean up code after you write it
%        1. Respect this line -------------------------------------------->
%   3. document!
%        a. cells
%        b. help text
%        c. comments
%   4. Vectorize when possible, and use built-in functions where possible
%   5. Choose consistent naming conventions

% Examples:  poor.m and better.m 

%% Figures

close all;clear;clc;
% You can create a figure (graphics window) with the FIGURE command:
figure;

% MATLAB uses figure handles to refer to these windows
figHandle=figure;

% Example of using handles: we can set properties of an existing figure:
set(figHandle, 'Position', [0 0 400 500])
% [x0,y0,width, height], x0,y0 are the coordinates of the bottom left corner

% figure can have arguments
figure('Position',[0 0 400 500]) 

figure('Name','An Empty Graphics Window')

% General syntax: 
% figure('PropertyName',value,...) lets you describe properties

% An entire list of figure properties is available in Matlab documentation.
docsearch 'figure properties'

% You can also adjust things interactively
figure
plot(0:0.1:pi,sin(0:0.1:pi))
propertyeditor

% Useful commands:
figHandle=gcf;  % returns the current figure handle. 
                % If a figure does not exist, then gcf creates a figure 
                % and returns its handle.
shg % shows most recent graphic window
clf % clears the current figure
close(figHandle)
close(gcf)
close all % closes all figures
%% 2D Plotting

%% The plot function
close all;clear;clc;

% Very basic plotting
x = linspace(0,10,100); % linspace(x1,x2,n) generates n points. 
                        % The spacing between the points is (x2-x1)/(n-1).
                        % linspace(x1,x2) assumes n=100.
y = cos(x);
figure
plot(x,y);

% if x is not provided, index values are used: 1,2,...,length(y).
plot(sin(1:100))  % overwrites
                  % use 'hold on' to plot sin on top of cos

% For plotting styles
docsearch LineSpec
plot(x,y,'--');
plot(x,y,'x');
plot(x,y,'m');
plot(x,y,'--xm');

% A bit more complicated example: 
plot(x,y,'ks','MarkerSize',4); % k - "black", s - "square"
% For more,
docsearch LineSpec

% We can plot multiple data sets or curves in one figure:
x2 = linspace(0,10,1000);
y2 = sin(x2);
plot(x,y,'r','LineWidth',5);
hold on;  % prevents overwriting
plot(x2,y2,'b','LineWidth',7);

% Matrices are interpreted columnwise
A=repmat((1:10)',1,5);  % 10-by-5 matrix
B=rand(10,5);           % 10-by-5 matrix 
plot(A,B);  % plots 5 sets of 10 points

% repmat above is not necessary: we can plot many y's vs the same x's
X=[1:10]';
Y=rand(10,5);
plot(X,Y); 

% By default, plot commands draw to the current figure,
% overwriting anything there
plot(rand(1,10)); 
plot(rand(1,10)); 

% the hold on/off command alters this behavior:
hold on
plot(rand(1,10)); 
plot(rand(1,10)); 
hold off
plot(rand(1,10)); 

% For computations that update a graph at every step, use the command 
% "drawnow" afer a "plot" command to force the graph to refresh, otherwise 
% Matlab might wait to draw the graph until all the computation is done.  
for i=1:1000
    plot(cos(i/1000 + [1:.01:10]),'r');
    drawnow;  % Use this command if you want to modify graphics objects 
              % and want to see the updates on the screen immediately.
end

% Colors can be specified in many ways
docsearch ColorSpec
plot(rand(1,10),'Color',[1 0 0]);  % RGB
plot(rand(1,10),'Color','g'); 
plot(rand(1,10),'Color','blue'); 

% Axis:
axis([-1 1 -2 2])
%[xmin, xmax, ymin, ymax]
axis off  
axis on     % by default, [0,1]*[0,1]
axis equal  % Use the same length for the data units along each axis.
axis square % Use axis lines with equal lengths. 
            % Adjust the increments between data units accordingly.
axis ij     % the y-axis is vertical with values increasing from top to bottom.
axis xy     % back to usual
axis image  % Use the same length for the data units along each axis and fit 
            % the axes box tightly around the data.
axis normal % Restore the default behavior.
 
% A few useful commands:
xlim([0 10])
grid on
xlabel('x');
ylabel('y');
title('my title')
