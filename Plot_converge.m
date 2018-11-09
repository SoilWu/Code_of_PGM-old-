%% **************************************************************
%  filename: Plot_converge
%% ***************************************************************
%% to plot the convergence figure with synthetic data
%%
%% Copyright by Shaohua Pan and Wuyu Qia, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"


addpath('D:\Wu_Yuqia\test_SPCA\solvers')

p = 500;

k = 10; 

ns = 50;

d = [400;300;ones(p-2,1)];
        
D = diag(d);

temp_Sigma = randn(p);
        
temp_Sigma = (temp_Sigma + temp_Sigma')/2;  % do not forget the symmetrization !!!!
        
[P,~] = eig(temp_Sigma);

P(:,1:2) = zeros(p,2);        
        
w = 1/sqrt(10)*ones(k,1);
       
P(1:k,1) = w;
        
P(k+1:2*k,2) = w;
        
v1 = P(:,1);
        
v2 = P(:,2);
        
Sigma = P*D*P';
        
mu = zeros(1,p);
        
X = mvnrnd(mu,Sigma,ns);
        
A = X'*X;

[~,~,iter,xiter_list,fobj_list] = PGM_SPCA(A,0.001);

n1 = size(xiter_list,2);

diff_obj = fobj_list - fobj_list(1);

temp_mat = xiter_list - repmat(xiter_list(:,1),[1,n1]);

diff_xnorm = sum(temp_mat.*temp_mat,1).^(1/2);

%% ***************** Plot figure *********************************
subplot(1,2,1);
plot([1:iter],flip(diff_xnorm),'r--*','LineWidth',2);
xlabel('Iteration')
ylabel('||x^k-x*||')
legend('PGM')
hold on;
subplot(1,2,2)
plot([1:iter],flip(diff_obj),'b--*','LineWidth',2)
xlabel('Iteration')
ylabel('\Phi_{\nu}(x^k)-\Phi_{\nu}(x*)')
legend('PGM')



