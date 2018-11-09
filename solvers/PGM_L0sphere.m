%% *************************************************************
% Filename: PGM_L0sphere
%% ************************************************************* 
% Solve the following nonconvex composite problem with proximal gradient method
% 
%    min f(x) + lambda ||x||_0 + 0.5*gamma ||x-xk||^2: s.t. ||x||=1    £¨*£©
%
%  Notice that the problem in (*) is equivalent to 
%  
%   min  f(x) + 0.5*gamma ||x-xk||^2 + lambda h(x) 
%
%  where h(x)=||x||_0 + delta_S(x), and S denotes the unit sphere.
%
%  In each step of PGM, the following optimization problem is solved
%
%  min (tau_gamma/2)||x - gk||^2 + lambda h(x)
% 
%  where gk = x^k - (1/tau_gamma) gradf(x^k), tau_gamma = tau+gamma
%% *************************************************************
%%
%% ***************************************************************
%% Copyright by Shaohua Pan and Wuyu Qia, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"
%% ***************************************************************

%% 
%% 
 function [xopt,loss,iter,xiter_list,fobj_list] = PGM_L0sphere(x,A,OPTIONS,lambda,gamma)
 %%
 if isfield(OPTIONS,'maxiter');          maxiter    = OPTIONS.maxiter;    end
 if isfield(OPTIONS,'printyes');         printyes   = OPTIONS.printyes;   end
 if isfield(OPTIONS,'Lipconst');         Lipconst   = OPTIONS.Lipconst;   end
 if isfield(OPTIONS,'tol');              tol        = OPTIONS.tol;        end

%%
 if (printyes)
    fprintf('\n *****************************************************');
    fprintf('******************************************');
    fprintf('\n ************* PGM for the sphere constrained L0 regularized minimization **************');
    fprintf('\n ****************************************************');
    fprintf('*******************************************');
    fprintf('\n  iter   optmeasure     diff_obj     norm_gdiff     gamma      tau   lambda   time');
 end

%% ********************** set the value of tau ***********************

tau = Lipconst;

tau_gamma = tau + gamma;

%%
%% ***********************  Main Loop ***************************

tstart = clock;

xiter_list = [];

fobj_list = [];

Ax = A*x;  loss_old = 0;

for iter = 1:maxiter
    
    gk = x -(2/tau_gamma)*Ax;
    
    xnew = Prox_L0sphere(gk,lambda/tau_gamma);

    abs_xnew = abs(xnew);
    
    xznorm = sum(abs_xnew>1.0e-8*max(abs_xnew));
    
    Axnew = A*xnew;
    
    loss = xnew'*Axnew;

    if nargout>3
        
        xiter_list = [xnew  xiter_list];
           
        fobj_list =[loss+lambda*xznorm fobj_list];
        
    end
       
    ttime = etime(clock,tstart);
        
    %% ************** generate the new iterate xnew ***************
    
    xdiff = xnew - x;   
    
    grad_diff = Axnew - Ax;
            
    optcond = 2*grad_diff - tau_gamma*xdiff;
    
    scale = tau*max(1,norm(x));
    
    opt_measure = norm(optcond)/scale;
    
    norm_gdiff = norm(grad_diff)/scale;
            
    diff_obj = loss - loss_old; 
           
    if (printyes)&&(mod(iter,1)==0)
        
         fprintf('\n %3d    %3.2e     %3.2e     %3.2e      %3.2e   %3.1f    %3.2e    %3.1f',iter,opt_measure,diff_obj,norm_gdiff,gamma,tau,lambda,ttime);
        
    end
    %%
    %% ************* check stopping criterion ******************
    %%
    if (opt_measure<tol)
        
        xopt = x;
        
        return;
    end
    
    if (gamma~=0)
        
        gamma = max(1.0e-8,min(0.5*gamma,1.0e-5*abs(loss)));
        
        tau_gamma = tau + gamma;
    end
    
    x = xnew;  
    
    Ax = Axnew;  
    
    loss_old = loss;
    
end

if (iter==maxiter)
    
    xopt = x;
    
end

