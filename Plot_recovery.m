%% **************************************************************
%  filename: Plot_recovery
%% ***************************************************************
%% to plot the sucessful recovery of PGM and GPower_l0 
%% 
%% Copyright by Shaohua Pan and Wuyu Qia, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"

addpath(genpath(pwd));

nexample = 500;

dim = 500;

ns = [50:10:540];

m = length(ns);

PGM_sratio = zeros(m);

GPM_sratio = zeros(m);

k = 10;

rho = 0.1;

PGM_result1 = zeros(nexample,m);

PGM_result2 = zeros(nexample,m);

GPM_result1 = zeros(nexample,m);

GPM_result2 = zeros(nexample,m);

for j=1:m
    
    for i = 1:nexample
        
        randstate = (j-1)*nexample+i
        
        randn('state',double(randstate));
        
        rand('state',double(randstate));
        
        d = [400;300;ones(dim-2,1)];
        
        D = diag(d);
        
        w = 1/sqrt(10)*ones(k,1);
        
        temp_Sigma = randn(dim,dim);
        
        temp_Sigma = (temp_Sigma + temp_Sigma')/2;  % do not forget the symmetrization !!!!
        
        [P,~] = eig(temp_Sigma);   
        
        P(:,1:2) = zeros(dim,2);
        
        P(1:k,1) = w;
        
        P(k+1:2*k,2) = w;
        
        v1 = P(:,1);
        
        v2 = P(:,2);
        
        Sigma = P*D*P';
        
        mu = zeros(1,dim);
        
        X = mvnrnd(mu,Sigma,ns(j));
        
        A = X'*X;
        
        [xopt1,~,iter1] = PGM_SPCA(A,rho);
        
        PGM_result1(j,i) = abs(xopt1'*v1);
        
        xopt1A = xopt1'*A;
        
        A1 = A-(A*xopt1)*xopt1'- xopt1*xopt1A +(xopt1A*xopt1)*(xopt1*xopt1');
        
        [xopt2,~,iter2] = PGM_SPCA(A1,rho);
        
        PGM_result2(j,i) = abs(xopt2'*v2);
        
      %% ************* Generalized power method ********************
        
        yopt1 = GPower(A,1/4,1,'l0',0);
        
        GPM_result1(j,i) = abs(yopt1'*v1);
        
        yopt1A =yopt1'*A;
        
        A1 = A-(A*yopt1)*yopt1'- yopt1*yopt1A +(yopt1A*yopt1)*(yopt1*yopt1');
        
        yopt2 = GPower(A1,1/4,1,'l0',0);
        
        GPM_result2(j,i) = abs(yopt2'*v2);
        
    end
    
    PGM_sratio1(j) = length(find(PGM_result1(j,:)>0.99))/nexample;
    
    PGM_sratio2(j) = length(find(PGM_result2(j,:)>0.99))/nexample;
 
    GPM_sratio1(j) = length(find(GPM_result1(j,:)>0.99))/nexample;
    
    GPM_sratio2(j) = length(find(GPM_result2(j,:)>0.99))/nexample;
end

save('recover_result','PGM_sratio1','PGM_sratio2','GPM_sratio1','GPM_sratio2');

subplot(1,2,1);
h1=plot(ns,PGM_sratio1,'r-o',ns, GPM_sratio1, 'b-');  
set(h1,'LineWidth',2.5) 
xlabel('(a) Size of sample');   ylabel('Recoverability');
set(get(gca,'XLabel'),'FontSize',10);
set(get(gca,'YLabel'),'FontSize',10);
legend('PGM','GPower\_l0');
hold on;
subplot(1,2,2);
h2=plot(ns,PGM_sratio2,'r-o',ns, GPM_sratio2, 'b-');  
set(h2,'LineWidth',2.5) 
xlabel('(b) Size of sample');   ylabel('Recoverability');
set(get(gca,'XLabel'),'FontSize',10);
set(get(gca,'YLabel'),'FontSize',10);
legend('PGM','GPower\_l0');
hold on;
