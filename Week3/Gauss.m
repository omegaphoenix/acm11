function x =Gauss(A,b)

[L,U,P] = lu(A); % PLU factorization  PA=LU
b1=P*b;          % PAx=Pb -> LUx=b1, Ux=y 
y=L\b1;          % "forward" substitution
x=U\y;           % "backward" substitution

end