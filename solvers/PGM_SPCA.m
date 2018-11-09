%% ***************************************************************
%% filename: PGM_SPCA
%%
%% ***************************************************************
%%
%%  max x'*A*x - nu ||x||_0 s.t. ||x||=1
%%
%%  where A is a positive semidefinite matrix
%% **************************************************************
%% Copyright by Shaohua Pan and Wuyu Qia, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"


function [xopt,variance,iter,xiter_list,fobj_list] = PGM_SPCA(A,rho)

n = size(A,1);

%%
%% ***************** to estimate ||A|| ***************************
%%
options.tol = 1e-6;
options.issym = 1;
options.disp  = 0;
options.v0 = randn(n,1);
[xint,Asnorm] =eigs(@(y)(A*y),n,1,'LM',options);

%% *********** parameters for PGM with extrapolation **************

OPTIONS_PGM.tol = 1.0e-6;

OPTIONS_PGM.printyes = 0;

OPTIONS_PGM.Lipconst = 2.0001*Asnorm;

OPTIONS_PGM.maxiter = 3000;

gamma = 0;    % or 1.0e-3*Asnorm

lambda = rho*Asnorm; 

if nargout>3

    [xopt,loss,iter,xiter_list,fobj_list] = PGM_L0sphere(xint,-A,OPTIONS_PGM,lambda,gamma);

else
    
    [xopt,loss,iter] = PGM_L0sphere(xint,-A,OPTIONS_PGM,lambda,gamma);

end

variance = abs(loss);

