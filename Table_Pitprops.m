%% *************************************************************
% filename: Table_Pitprops
%% *************************************************************
%% 
%% Copyright by Shaohua Pan and Wuyu Qia, 2018/11/8
%  our paper: "A globally and linearly convergent PGM for zero-norm 
%  regularized quadratic optimization with sphere constraint"
%%

addpath(genpath(pwd));

% A =  ([[1, 0.954, 0.364, 0.342, -0.129, 0.313, 0.496, 0.424, 0.592, 0.545, 0.084, -0.019, 0.134],
%          [0.954, 1, 0.297, 0.284, -0.118, 0.291, 0.503, 0.419, 0.648, 0.569, 0.076, -0.036, 0.144],
%          [0.364, 0.297, 1, 0.882, -0.148, 0.153, -0.029, -0.054, 0.125, -0.081, 0.162, 0.220, 0.126],
%          [0.342, 0.284, 0.882, 1, 0.220, 0.381, 0.174, -0.059, 0.137, -0.014, 0.097, 0.169, 0.015],
%          [-0.129, -0.118, -0.148, 0.220, 1, 0.364, 0.296, 0.004, -0.039, 0.037, -0.091, -0.145, -0.208],
%          [0.313, 0.291, 0.153, 0.381, 0.364, 1, 0.813, 0.090, 0.211, 0.274, -0.036, 0.024, -0.329],
%          [0.496, 0.503,-0.029, 0.174, 0.296, 0.813, 1, 0.372, 0.465, 0.679, -0.113, -0.232, -0.424],
%          [0.424, 0.419, -0.054, -0.059, 0.004, 0.090, 0.372, 1, 0.482, 0.557, 0.061, -0.357, -0.202],
%          [0.592, 0.648, 0.125, 0.137, -0.039, 0.211, 0.465, 0.482, 1,  0.562, 0.085, -0.127, -0.076],
%          [0.545, 0.569, -0.081, -0.014, 0.037, 0.274, 0.697, 0.557, 0.526, 1, -0.319, -0.368, -0.291],
%          [0.084, 0.076, 0.162, 0.097, -0.091, -0.036, -0.113, 0.061, 0.085, -0.319, 1, 0.029, 0.007],
%          [-0.019, -0.036, 0.220, 0.169, -0.145, 0.024, -0.232, -0.357, -0.127, -0.368, 0.029, 1, 0.184],
%          [0.134, 0.144, 0.126,  0.015, -0.208, -0.329, -0.424, -0.202, -0.076, -0.291, 0.007, 0.184, 1]]) ;

A= ([[1, 0.954, 0.364, 0.342, -0.129, 0.313, 0.496, 0.424, 0.592, 0.545, 0.084, -0.019, 0.134],
         [0.954, 1, 0.297, 0.284, -0.118, 0.291, 0.503, 0.419, 0.648, 0.569, 0.076, -0.036, 0.144],
         [0.364, 0.297, 1, 0.882, -0.148, 0.153, -0.029, -0.054, 0.125, -0.081, 0.162, 0.220, 0.126],
         [0.342, 0.284, 0.882, 1, 0.220, 0.381, 0.174, -0.059, 0.137, -0.014, 0.097, 0.169, 0.015],
         [-0.129, -0.118, -0.148, 0.220, 1, 0.364, 0.296, 0.004, -0.039, 0.037, -0.091, -0.145, -0.208],
         [0.313, 0.291, 0.153, 0.381, 0.364, 1, 0.813, 0.090, 0.211, 0.274, -0.036, 0.024, -0.329],
         [0.496, 0.503,-0.029, 0.174, 0.296, 0.813, 1, 0.372, 0.465, 0.679, -0.113, -0.232, -0.424],
         [0.424, 0.419, -0.054, -0.059, 0.004, 0.090, 0.372, 1, 0.482, 0.557, 0.061, -0.357, -0.202],
         [0.592, 0.648, 0.125, 0.137, -0.039, 0.211, 0.465, 0.482, 1,  0.526, 0.085, -0.127, -0.076],
         [0.545, 0.569, -0.081, -0.014, 0.037, 0.274, 0.679, 0.557, 0.526, 1, -0.319, -0.368, -0.291],
         [0.084, 0.076, 0.162, 0.097, -0.091, -0.036, -0.113, 0.061, 0.085, -0.319, 1, 0.029, 0.007],
         [-0.019, -0.036, 0.220, 0.169, -0.145, 0.024, -0.232, -0.357, -0.127, -0.368, 0.029, 1, 0.184],
         [0.134, 0.144, 0.126,  0.015, -0.208, -0.329, -0.424, -0.202, -0.076, -0.291, 0.007, 0.184, 1]]);

if (norm(A-A','fro')>1.0e-12)
    display('A is not a covariance matrix')
    return;
end

n = size(A,1);

k = 6;

[P D] = eig(A);

X1 = P*diag(diag(D).^(1/2))*P';

rho = 0.1;

rho1 = 0.15;

PGM_eigvec = zeros(n,k);

PGM_nzeigv = zeros(k,1);

PGM_variance = zeros(k,1);

GPM_eigvec = zeros(n,k);

GPM_nzeigv = zeros(k,1);

GPM_variance = zeros(k,1);

%% *******************The first one *****************************

[xopt1,PGM_variance(1),iter1] = PGM_SPCA(A,rho);

PGM_eigvec(:,1) = xopt1;

PGM_nzeigv(1) = sum(abs(xopt1)>1.0e-8*max(abs(xopt1)));

%%
yopt1 = GPower(A,rho1,1,'l0',0);

GPM_eigvec(:,1) = yopt1;

GPM_nzeigv(1) = sum(abs(yopt1)>1.0e-8*max(abs(yopt1)));

GPM_variance(1)=yopt1'*A*yopt1;

%% ******************* The second one *****************************

xopt1A = xopt1'*A;

A1 = A-(A*xopt1)*xopt1'- xopt1*xopt1A +(xopt1A*xopt1)*(xopt1*xopt1');
 
[xopt2, PGM_variance(2),iter2] = PGM_SPCA(A1,rho);

PGM_eigvec(:,2) = xopt2;

PGM_nzeigv(2) = sum(abs(xopt2)>1.0e-8*max(abs(xopt2)));

%% 
yopt1A = yopt1'*A;

A1 = A-(A*yopt1)*yopt1'- yopt1*yopt1A +(yopt1A*yopt1)*(yopt1*yopt1');
 
yopt2 =  GPower(A1,rho1,1,'l0',0);

GPM_eigvec(:,2) = yopt2;

GPM_nzeigv(2) = sum(abs(yopt2)>1.0e-8*max(abs(yopt2)));

GPM_variance(2)=yopt2'*A1*yopt2;

%% ******************* The third one *****************************

xopt2A = xopt2'*A1;

A2 = A1-(A1*xopt2)*xopt2'- xopt2*xopt2A +(xopt2A*xopt2)*(xopt2*xopt2');
 
[xopt3, PGM_variance(3),iter3] = PGM_SPCA(A2,rho);

PGM_eigvec(:,3) = xopt3;

PGM_nzeigv(3) = sum(abs(xopt3)>1.0e-8*max(abs(xopt3)));

%%
yopt2A = yopt2'*A1;

A2 = A1-(A1*yopt2)*yopt2'- yopt2*yopt2A +(yopt2A*yopt2)*(yopt2*yopt2');
 
yopt3 = GPower(A2,rho1,1,'l0',0);

GPM_eigvec(:,3) = yopt3;

GPM_nzeigv(3) = sum(abs(yopt3)>1.0e-8*max(abs(yopt3)));

GPM_variance(3)=yopt3'*A2*yopt3;

%% ******************* The fourth one *****************************

xopt3A = xopt3'*A2;

A3 = A2-(A2*xopt3)*xopt3'- xopt3*xopt3A +(xopt3A*xopt3)*(xopt3*xopt3');
 
[xopt4,PGM_variance(4),iter4] = PGM_SPCA(A3,rho);

PGM_eigvec(:,4) = xopt4;

PGM_nzeigv(4) = sum(abs(xopt4)>1.0e-8*max(abs(xopt4)));

%%
yopt3A = yopt3'*A2;

A3 = A2-(A2*yopt3)*yopt3'- yopt3*yopt3A +(yopt3A*yopt3)*(yopt3*yopt3');
 
yopt4 = GPower(A3,rho1,1,'l0',0);

GPM_eigvec(:,4) = yopt4;

GPM_nzeigv(4) = sum(abs(yopt4)>1.0e-8*max(abs(yopt4)));

GPM_variance(4)=yopt4'*A3*yopt4;


%% ******************* The fifth one *****************************

xopt4A = xopt4'*A3;

A4 = A3-(A3*xopt4)*xopt4'- xopt4*xopt4A +(xopt4A*xopt4)*(xopt4*xopt4');

[xopt5, PGM_variance(5),iter5] = PGM_SPCA(A4,rho);

PGM_eigvec(:,5) = xopt5;

PGM_nzeigv(5)=sum(abs(xopt5)>1.0e-8*max(abs(xopt5)));

%%
yopt4A = yopt4'*A3;

A4 = A3-(A3*yopt4)*yopt4'- yopt4*yopt4A +(yopt4A*yopt4)*(yopt4*yopt4');
 
yopt5 =  GPower(A4,rho1,1,'l0',0);

GPM_eigvec(:,5) = yopt5;

GPM_nzeigv(5)=sum(abs(yopt5)>1.0e-8*max(abs(yopt5)));

GPM_variance(5)=yopt5'*A4*yopt5;
%% ******************* The sixth one *****************************

xopt5A = xopt5'*A4;

A5 = A4-(A4*xopt5)*xopt5'- xopt5*xopt5A +(xopt5A*xopt5)*(xopt5*xopt5');

[xopt6,PGM_variance(6),iter6] = PGM_SPCA(A5,rho);

PGM_eigvec(:,6) = xopt6;

PGM_nzeigv(6)=sum(abs(xopt6)>1.0e-8*max(abs(xopt6)));

%%
yopt5A = yopt5'*A4;

A5 = A4-(A4*yopt5)*yopt5'- yopt5*yopt5A +(yopt5A*yopt5)*(yopt5*yopt5');
 
yopt6 = GPower(A5,rho1,1,'l0',0);

GPM_eigvec(:,6) = yopt6;

GPM_nzeigv(6) = sum(abs(yopt6)>1.0e-8*max(abs(yopt6)));

GPM_variance(6)=yopt6'*A5*yopt6;

PGM_eigvec
PGM_nzeigv
PGM_variance/13
GPM_eigvec
GPM_nzeigv
GPM_variance/13
%% *********** final result ************************************

Z = X*[xopt1  xopt2  xopt3  xopt4  xopt5   xopt6];
[Q, R] = qr(Z);
r = diag(R);
PGM_IVar = zeros(6,1);

for i=1:6
    
   PGM_IVar(i) = sum(norm(r(1:i)).^2)/13;
end
PGM_IVar
PGM_IVar(2:end)-PGM_IVar(1:end-1)

 
Z = X*[yopt1  yopt2  yopt3  yopt4  yopt5   yopt6];
[QQ, RR] = qr(Z);
rr = diag(RR);
GPM_IVar = zeros(6,1);

for i=1:6
    
   GPM_IVar(i) = sum(norm(rr(1:i)).^2)/13;
end
GPM_IVar
GPM_IVar(2:end)-GPM_IVar(1:end-1)

