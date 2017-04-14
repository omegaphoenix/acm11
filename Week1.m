% ACM 11: Introduction to MATLAB   

%% Week 1: Introduction, Basic Syntax, Help System, Vectors, (Sparse) Matrices, Logical Operations

%% What is MATLAB?

%{
MATLAB is short for MATrix LABoratory. It is a numerical computation 
environment which provides a suite of tools for computation, visualization,
and more. MATLAB excels at numerical computation.

When to use MATLAB?
(1) if you can afford to buy it, or can get it free (like here at Caltech).
(2) for numerically intensive computations.
(3) for plotting and dealing with data (analisis, visualization).
(4) for faster DEVELOPMENT than C and Fortran.
(5) for faster EXECUTION than most other high level languages.

When not to use MATLAB?
(1) when you have to pay for it.
(2) for symbolic math, if you have access to Mathematica or Maple.
(3) if you want free software, or want to release code that doesn't
require the user to have MATLAB.
(4) if you are doing something really serious, like climate modelling, 
solving 3D Partial Differential Equations, working with massive datasets, etc.

MATLAB competes with many languages, including:
- Mathematica and Maple  [sybolic math packages]
- MATLAB imitations [FreeMat, Octave, SciLab, xmath]
- C and Fortran   [staples of the high-performance community]
- C++, as well as C# etc.   [modern Object-Oriented languages]
- java and python   [also modern Object-Oriented languages]
- perl, bash and other shells   [mainly data manipulation]
- S and R [statistics languages]

In one way or another, most of MATLAB's competitors can do what MATLAB 
does. Here are some of MATLAB's advantages:

(1) the language is intuitive and mathematically expressive =>
    MATLAB is rather fast to learn
(2) the HELP menu is fantastic
(3) MATLAB is an industry standard (much like Microsoft Word) =>
    the web is full of MATLAB resources.
(4) MATLAB matrix manipulation algorithms (esp. for sparse matrices) are 
state of the art

and disadvantages:

(1) for complex tasks (especially ones which require for loops), MATLAB
can sometimes be slower than hand-coded C or Fortran.
(2) MATLAB is expensive 
%}

%% Getting started
%{
Elements of the Matlab desktop: 

  - command window, command prompt (command line interpreter)
  - history window (history features are also available at the 
    command prompt: <up> key, tab completion, drag-n-drop)
  - workspace, variable editor (and plot window)
  - current directory and file details editor
  - the current folder selector, which help determines search path (make
    sure you're consistent about which directory you work in)
  - the editor 

All of these windows can be re-arranged
%}

%{ 
What is this file?

This is an example of a script "m-file". All script files have the 
extension ".m" This file is saved as "Week1.m". MATLAB also automatically 
makes backup files, called either "Week1.asv" (windows) or "week1.m~" 
(unix/linux). You can ignore these, or turn them off in the Preferences.

Script m-files contain sequences of commands that are executed in order.

%}

%{
Two important concepts:

(1) Comments. Everything written here is a comment,
text that is ignored by  MATLAB. Use comments to document your code!  
There are two ways to write comments: either comment a single line with 
"%", or use block comments to comment several lines, using 
%{ followedd by %}. See this file for examples. Note that you cannot have 
ANYTHING on the line that starts with "%{". By default, the editor puts 
comments in green (you can change this). 

(2) Cells.  MATLAB has two meanings for the word "cell". 
The first meaning of "cell" is used in a m-file, and it is a way to split 
the m-file into chunks, each of which can be executed separately.  
This is a very useful tool.  To execute a single cell, see the button in 
the toolbar, or  hold down "ctrl" and then hit "enter" on your keyboard.
The beginning of a cell is a special comment line that  begins with two %%, 
not just one %.  The other use of the word "cell" refers to a type of 
variable, and will be discussed later.

We will use cells a lot as a convenient way to step through code.
%}

%{
Other basics:

The following shortcuts are useful: 
(1) saving the file: ctrl+S for windows, ctrl+S for unix. 
(2) commenting a line of code: ctrl+R for windows, ctrl+/ for unix. 
(3) uncommenting a line of code: ctrl+T for unix and windows
        
%}

% Note: to execute a subset of lines, you can enclose them in a cell, 
% or highlight them and use "Evaluate Selection" (right click)

%% The Help System
%{ 
 Help shows the format(s) for using a command and directs you to related
 commands; without any arguments, it gives you a hyperlinked list of
 topics to find help on; with a topic as an argument, it gives you a list
 of subtopics
%}

help plot
help qr
help   

% doc is like help, except it comes up in a different window, and may
% include more details
help fft
doc fft

% lookfor is used when you don't know what command you want; it does 
% something like a keyword search through the documentation.

lookfor wavelet % find MATLAB's matrix decomposition functions

% similarly, you can use docsearch
docsearch fourier

% demo gives video guides and example code
demo

%% More sources of help
%{
    The internet has about 10,000 MATLAB tutorials.  
%}
%% MATLAB as a Calculator
% standard commands to start a session:
clc       % clears the Command Window screen
clear     % clears all variable definitions
close all % closes all figures

3 + 5  % displays result

3 + 5; % suppresses display of result (but the calculation is done)

3 +        5 % white space doesn't matter where it shouldn't

% Wrapping lines
3 + ...
5

% as long as i or j hasn't been assigned to as a regular variable, MATLAB
% interprets them as sqrt(-1)
i^2
exp(2*i*pi)
1 + j 

% If you set
i=1
% then i is 1 now, but if you clear its value:
clear i
% it is sqrt(-1) again
disp(i)
% Note: to clear more than one variable, do NOT separate the variable
% names with a comma -- just uses spaces, e.g.
% >> clear x y z

% MATLAB respects the standard order of operations
3 + 5/2 
(3 + 5)/2

% Note: comments can start ANYWHERE in a line of code
% but you cannot "undo" a comment on the same line, e.g. 
% comment % code

%% Variables
% MATLAB doesn't require you to declare variables, and is case sensitive.
x=4
disp(X) % gives error (disp(X) asks to show the value of X)
% Variable name must start with a upper case or lower case letter and be
% followed by letters, digits and underscores ONLY. Variable name cannot be
% the same as a built in MATLAB keyword (use iskeyword to check)

a = 2
A = 3
a^A
% MATLAB stores the result of the last calculation as a variable named 'ans'
ans+1  

x = 8; a+x % the semicolon suppresses output AND allows
           % to add a second command.
           
%  who and whos give information on the variables currently defined
%  you can also see this information in the 'workspace' tab, which
%  lists all the global data you have in memory
who 
whos

clear A a ans % get rid of "A" "a" and "ans"
who
           
%% Vectors and Matrices
% Matrices are the basic datatype in Matlab
clc; clear;
x = [1, 2, 3] % row vector: separate elements in the same row with commas
y = [1 2 3] % row vector: spaces also work to separate elements in a row
z = [1; 2; 3] % column vector: separate elements in the same column with ;

z = 1:20    % can use the range notation to generate row vectors
z = 1:2:20  % the "2" means that we go from 1 to 20 in increments of 2
z = 20:-2:0
z = [0:.1:1] % the array brackets are optional

% a 3-by-3 matrix: 
A = [1, 2, 3; 4, 5, 6; 7, 8, 9] 
%   the spaces and commas separate columns
%   and the semicolons separate rows

% Another (more visual) way to make a matrix
A = [1 2 3 
    4 5 6 
    7 8 9]
%   in this case, the second line indicates a new row


%A*x % invalid! Lust like in linear algebra, quantities need to have 
     % the correct dimensions to make a valid MATLAB expression
x*A  % this works: row vector times matrix is a row vector     

z = [1; 2; 3]
A*z % this also works: matrix times column vector yields column vector

% some operations are "matrix" operations, and others  are "component-wise" 
% aka "element-wise"

B=A*A   % this is matrix  multiplication
C=A.*A  % this works element-wise

BB=A^2     % this is matrix multiplication, e.g. "A*A"
CC=A.^2    % this is element-wise, e.g. "A.*A"

exp(A)      % this is element-wise
expm(A)     % this is an inherent matrix operation

% use A' to take the conjugate transpose of a matrix, 
% and A.' to take the real transpose; e.g.: 

A = [0, i; -i, 0]
A.'
A'

% For example, you can define column vectors like this
x = [1, 5, 6, 7, 7].'

% Matlab requires you to include * explicitly for multiplication
a = 1;
b = 2;
a * b 
%a b is an error

%% More on Matrices
% Practically every task in MATLAB uses matrices in some way, so let's
% go over some more ways to work with them.
A = [1 4 7; 2 5 8; 3 6 9]

% Indices: use ( ) to access an element, not [ ]
% For creating a matrix, use [ ] not ( ).

% reading
A(1,1) %indexing is 1-based in MATLAB 
A(1,2)
A([1,3],1)
A(2,[2,3])
A(1:3,1)
A(:,1)
A(1,:)
A([1,3,2],:) % permutes rows
A([1,3,2],[1 3 2]) % permutes rows and columns

% getting the size
length([1,2,3]) %only for row/column vectors
[m,n] = size(A)

% writing
A(1,1) = 3
A(1:3,2:3) = NaN % not a number
A(1:2,1:3) = [100 200 300; 400 500 600]

%  Linear Indexing:
%  MATLAB uses "column major order", meaning a matrix is stored as
%  as a linear array, one column after another
%A(1:n+1:n*n) gives diagonals of n x n matirx
%A(1:n+1:end) gives diagonals of n x n matirx
A(:)
A(1:4:end) % handy trick to extract the diagonal: 
% if A is an N*N, then, A(1:N+1:end)
% the special keyword "end" refers to the last entry

% reshaping
reshape(1:9,3,3)   % reshapes array
B = [1 2 3 4; 5 6 7 8]
reshape(B,4,2)   % 4-by-2, or
reshape(B,4,[])  % with 4 rows
reshape(B,[],4)  % with 4 columns

% higher dimensional arrays
M = reshape(1:27,3,3,3)

% generating standard matrices
eye(5)    %the 'eye'dentity
ones(5)   % the same as ones(5,5)
ones(5,3)
rand(2,2)  % each entry is a sample from U(0,1)
randn(3,3) % each entry is a sample from N(0,1)
zeros(4,4)
inf(3,3) 
nan(4,1)

% repmat
M = repmat((1:3)',2,2) % repeats copies of an array

%  'block' operations ("generalization" on repmat)
A = [1 2; 3 4]
M = [A, 2*A; -A, 2*A]

%% Sparse Matrices
% In a few fields, the term "sparse" has become a buzzword.  What does it
% mean?  There's no quantitative definition, but the qualitative definition
% is easy: a vector or matrix is "sparse" when it has many zero entries.
% In Matlab, there are two ways to store a matrix.  The first way, which
% we've been using so far, is to store all entries.  If we know the
% dimensions of the matrix, and have a convention of how the entries are
% stored, then there is no need to store the indices.  The second method,
% which Matlab calls the "sparse" datatype, is to only store the nonzero
% entries.  But, with this scheme, we need to store the indices also, so
% there's a penalty in memory.  It is very important to realize that
% Matlab's definition of "sparse" only refers to the way a matrix is
% stored.  The same matrix can be sparse, or non-sparse (aka "full").

% Here's an example:
clc; clear;
A = randn(2)        % Method 1 of storing a matrix, aka "full" storage
S = sparse(A)       % Method 2 of storing a matrix, aka "sparse" storage
% The "sparse" function converts a matrix to the sparse storage format.  We
% can go the other direction using the "full" command, e.g.
% B = full(S);

% Now, suppose we have a matrix that really is sparse.  THEN the sparse
% matrix format has the advantage:
clear; 
A = zeros(100);
A(45,60) = 1;  % method 1
S = sparse(A)  % method 2
whos           % S uses less memory, as expected

% We can create a sparse matrix on its own, i.e. we don't have to convert
% from a full matrix first.  The simplest method is the following:
clear;
S = sparse(100,100) % makes an all zero 100 x 100 sparse matrix
% Note: S = sparse(100) is NOT what you want.  This would make a 1 x 1 sparse
% matrix, with the (1,1) entry being 100.

S(45,60) = 4

spy(S) % very useful for seeing the sparsity pattern. 

% sizes
[m,n]=size(S)
k=nnz(S) %[n]umber of [n]on-[z]eros

% Generating random sparse matrices
A=sprand(10,10,0.1) % random, 10-by-10, sparse matrix with approximately  
                    % 0.1*10*10 uniformly distributed nonzero entries 
B = sprandsym(1000,.01);    % this is a 1000 x 1000 sparse symmetric matrix
                            % approximately 1% of its entries are randomly 
                            % selected to be nonzero; on those entries, 
                            % the value is chosen from a uniform distribution.
spy(B); title('Sparsity pattern of the random symmetric matrix');

% This is more or less the way sparse matrices are constructed in practice:
iSet = [1 3 4 6];
jSet = [3 6 7 8];
vals = [122 342 345345 34534];
m=20;
n=20;
A=sparse(iSet, jSet, vals, m, n)

% If A is large, we can use spalloc to allocate space for A
clear; clc;
n = 10^5;
nonZeroCount = n*0.1;
tic
S = spalloc(n,n,nonZeroCount); % allocates space for sparse n-by-n matrix 
                               % with nonZeroCount nonzero entries.
for k=1:nonZeroCount
    i = ceil(n*rand);  % ceils rounds toward positive infinity
    j = ceil(n*rand);
    S(i,j) = rand; 
end
toc

% This is a more standard way (avoiding loops) to accomplish the same task
clear; 
n = 10^5;
nonZeroCount = n*0.1;
tic
S = sparse(...
    ceil(n*rand(1,nonZeroCount)),... % row indices
    ceil(n*rand(1,nonZeroCount)),... % col indices
    rand(1,nonZeroCount),...         % values
    n,n);              % dimensions, nnz
toc

%% Logical Operations

% zero = false, nonzero = true
%  "<=" , "==" , ">=" , "|" , "&" (vectorized)
a=[1, 2, 3, 4] <= [1, 3, 2, 1]
b=[1 0 1 1] | [1 0 0 0] % elementwise (vectorized) 'or'

% "||" , "&&" , and "~"  (not vectorized)
a=false;
b=true;
a || b % if a is true, returns true without looking at b
a && b % if a is false, returns false without looking at b
~a     % if a is true (false), returs false (true) 

% any and all
any([1 0 0 0]) % tests if ANY array elements are nonzero (true)
all([1 1 1 1]) % tests if ALL array elements are nonzero (true)
all([1 0 1 1])

% You can use logical values to perform 'logical indexing':
% instead of an array of integer indices, pass in a logical array
% Example:
A = [1 2 3; 4 5 6; 7 8 9]
A < 5 
A(A<5) = NaN % very handy!

%% Strings
% Strings are just arrays of char values in MATLAB. 
% Characters are essentially integers
clear; clc;
x='a string'
y=['a',' ','s','t','r','i','n','g']
char(77)
double('s')

% you can operate with strings accordingly
['this ', 'combines ', 'strings']

% ['this'; 'is an error because the dimensions are wrong']

['this is ok because the '; 'dimensions are the same']

% equality between strings
strcmp('aaa','bbbbb') % compares strings
strcmp('aaa','aaa')
 
% Some basic output functions
disp('display this string')
disp(3) % also displays numbers or other values

% fprintf: another way to print stuff to the screen, more advanced than disp
% Example:
time = datestr(now)    % get the time (this is a string)
fprintf('The time is: %s\n', time)
%  The "%s" is a placeholder for a % string.  
% Common place holders are:
%   %d      for integer
%   %f      for float (either single or double)
%   %e      for scientific notation
% most of these take additional options, like
%   %.6f    will print the number with 6 decimal places.
% There are also special characters, like \t for tab and \n for newline

% fprintf can be used to print to a file.  Here's how:
fileID = fopen('test.txt','wt'); % Open or create new file for writing.
fprintf(fileID, 'The time is: %s \nKeep calm, lunch time is coming', time); 
fclose(fileID);

%% Data types
% The default data type for a number in MATLAB is double precision floating
% point, usually just called a "double".  This uses 64 bits (e.g. 8 bytes),
% and it basically stores a number in scientific notation.
% Because of the bit-limitation, there are some numbers that are too large
% and too small to store.  We can find out what this range is:
clc; realmax
realmin
% The range is large!  For example, one estimate for the radius of the
% universe is 10^26 meters.  The radius of a proton is about 10^-15 meters,
% and the Planck Length is about 10^-25 meters.
% However, we still get problems easily.  Perhaps more important than the
% range is the machine epsilon.  This number is given in MATLAB by
eps
% A 'double' only has roughly 16 digits of precision.  This means that a
% number like
%   1.000000000000000000
% and
%   1.000000000000000001
% are the same to a computer.
% So, be careful!

%% Numeric Data Types
% these are the names of numeric data types, and also functions which
% convert to those data types:
%{
double % the workhorse
logical
single
int8
int16
int32
int64
uint8
uint16
uint32
uint64
char % not actually 'numeric', but behaves very similarly
%}

help datatypes

% tests
isnumeric(3)
isnumeric([1 2 3 4 5])
isnumeric('string')
isfloat(1)
isfloat(single(1))
isfloat(double(1))
isinteger(1) %<-- this is a double by default!
isinteger(int32(1))
isinteger(1.1)
islogical(1)
islogical(true)
[1 2 3 4] == [1 2 100 200] % but not what is output.

% floating point types have special values inf and nan
0/0
1e999^2
isnan(NaN)
isinf(Inf)
isfinite(NaN) % useful for automatically checking numerical code for problems
isfinite(-Inf)
isfinite(3)

% some other useful values for floating point numbers:
realmax('double')
realmax('single') 

