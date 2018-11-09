%% **************************************************************
%  filename: Plot_covariance
%% ***************************************************************
%% to plot the proportion of explained covariance for two solvers
%% PGM and GPower_l0
%%
%% Copyright by Shaohua Pan and Wuyu Qia, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"

addpath(genpath(pwd));

X = xlsread('cancer_data1.xlsx');

[m,n] = size(X);

Xt = X';

X = X - mean(Xt)'*ones(1,n);  % to such that each row of X has a zero mean

A = X*X';

[x0,~] = eigs(A,1);

variance = x0'*(A*x0);

P = linspace(0.0001, 0.01, 50);

Q = linspace(0.002, 0.4, 50);

x1 = zeros(50,1);

y1 = zeros(50,1);

x2 = zeros(50,1);

y2 = zeros(50,1);

for i = 1:50  
    
    [xopt,svariance,iter] = PGM_SPCA(A,P(i));
    
    x1(i) = sum(abs(xopt)>1.0e-8*abs(xopt));

    y1(i) = svariance/variance; 
end
    
for j = 1:50  
    
    xopt = GPower(A,Q(j),1,'l0',0);
    
    x2(j) = sum(abs(xopt)>1.0e-8*abs(xopt));
    
    y2(j) = xopt'*(A*xopt)/variance;
 
end

plot(x1,y1,'r-o',x2,y2,'b--','LineWidth',2);
xlabel('Cardinality');
ylabel('Proportion of explained variance(%)');
legend('PGM','GPower\_l0')


