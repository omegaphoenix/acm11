%%
i = 0;
for i = 1:1000
    if(i ~= 1000)  
        A(i,i+1) = 1; 
    end
end

%%
i = 0;
for i=1:1000
    if(i ~= 1) 
        A(i,i-1) = 1;
    end
end

%%
i = 0
for i=1:1000
   A(i,i) = 4;
end

%%
for i=1:1000
   b(i) = rand(1);  %  b=rand(1,1000) 
end

%%
x = lsqr(A,b')   
  
for i=1:1000
   err(i) = A(i,:)*x-b(i); 
end

max(max(max(abs(err))))  