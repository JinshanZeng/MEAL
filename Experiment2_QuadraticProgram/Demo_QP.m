% Ref: Jinshan Zeng, Wotao Yin and Ding-Xuan Zhou, Moreau envelope 
% augmented Lagrangian method for nonconvex optimization with linear
% constraints, Journal of Scientific Computing, 2021
% compare the performance of LiMEAL with Proximal iALM proposed in (Zhang&Luo, SIOPT 2020)
% for the following quadratic programming optimization with box constraints
% min_x 1/2 x^TQx+r^Tx  s.t. Ax=b, ell_i<=xi<=ui for all i

clear all; close all; clc;
%% settings of the quadratic programming
m = 5;
n = 20;
QP.Q = rand(n); % matrix Q involved in the objective
QP.r = rand(n,1); % vector r involved in the objective
QP.A = rand(m,n); % linear matrix of the constraints
xx = rand(n,1); % a feasible x to generate the vector b
QP.b = QP.A*xx; % vector b of the constraints
QP.ell = 0; % lower bound of box constraints (set the same for each coordinate)
QP.u = 1; % upper bound of box constraints
NormQ = norm(QP.Q); % the spectral norm of matrix Q

NumIter = 1e5; % number of iterations

%% Proximal inexact augmented Lagrangian method
x0 = rand(n,1);
z0 = x0;
lamb0 = zeros(m,1);
p = 2*NormQ; % inverse of proximal parameter > rho, the wealy convex modulus of objective
beta = 50; % penalty parameter in augmented Lagrangian
alpha = beta/4; % dual step size used in the update of multipliers
s = 1/(NormQ+p+beta*norm(QP.A)^2)/2; % primal step size
eta_alm1 = 1; % step size in z-update (reducing to inexact ALM)
[x_alm1,z_alm1,lamb_alm1,objfun_alm1,dualres_alm1,primalres_alm1]...
    = ProxiALM_QP(QP,x0,z0,lamb0,p,beta,alpha,s,eta_alm1,NumIter);

eta_alm2 = 0.5; % step size in z-update
[x_alm2,z_alm2,lamb_alm2,objfun_alm2,dualres_alm2,primalres_alm2]...
    = ProxiALM_QP(QP,x0,z0,lamb0,p,beta,alpha,s,eta_alm2,NumIter);

%% LiMEAL 
gamma = 1/2/NormQ; % proximal parameter (gamma in (0,1/||Q||))

eta1 = 0.5; % step size for z-step eta in (0,2) (when eta=1, LiMEAL reduces to the classical Prox-ALM)
[x1,z1,lamb1,objfun1,dualres1,primalres1,staterr1]=LiMEAL_QP(QP,x0,z0,lamb0,beta,gamma,eta1,NumIter);

eta2 = 1; % step size for z-step eta in (0,2) (when eta=1, LiMEAL reduces to the classical Prox-ALM)
[x2,z2,lamb2,objfun2,dualres2,primalres2,staterr2]=LiMEAL_QP(QP,x0,z0,lamb0,beta,gamma,eta2,NumIter);

eta3 = 1.5; % step size for z-step eta in (0,2) (when eta=1, LiMEAL reduces to the classical Prox-ALM)
[x3,z3,lamb3,objfun3,dualres3,primalres3,staterr3]=LiMEAL_QP(QP,x0,z0,lamb0,beta,gamma,eta3,NumIter);

% save objfun_alm1; save dualres_alm1; save primalres_alm1;
% save objfun_alm2; save dualres_alm2; save primalres_alm2;
% save objfun1; save dualres1; save primalres1; save staterr1;
% save objfun2; save dualres2; save primalres2; save staterr2;
% save objfun3; save dualres3; save primalres3; save staterr3;

Iter = (1:NumIter)';
% plot objective function
figure(1),
semilogx(Iter,objfun_alm1,'c-','LineWidth',2);
hold on;
semilogx(Iter,objfun_alm2,'m-','LineWidth',2);
hold on;
semilogx(Iter,objfun1,'b-','LineWidth',2);
hold on;
semilogx(Iter,objfun2,'r-','LineWidth',2);
hold on;
semilogx(Iter,objfun3,'k-','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('f(x^k)','FontSize',14);
legend('iALM','Prox-iALM','LiMEAL (\eta = 0.5)','LiMEAL (\eta = 1)','LiMEAL (\eta = 1.5)','FontSize',14);
% axes('Position',[0.18,0.72,0.12,0.09]); % magnify the detailed figures
% plot(Iter(4:6),objfun_alm1(4:6),'c-','LineWidth',2);
% hold on;
% plot(Iter(4:6),objfun_alm2(4:6),'m-','LineWidth',2);
% axis([4,6,32.95,33.13]);

% plot dual residual
figure(2),
loglog(Iter,dualres_alm1,'c-','LineWidth',2);
hold on;
loglog(Iter,dualres_alm2,'m-','LineWidth',2);
hold on;
loglog(Iter,dualres1,'b-','LineWidth',2);
hold on;
loglog(Iter,dualres2,'r-','LineWidth',2);
hold on;
loglog(Iter,dualres3,'k-','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('||Ax^k-b||','FontSize',14);
legend('iALM','Prox-iALM','LiMEAL (\eta = 0.5)','LiMEAL (\eta = 1)','LiMEAL (\eta = 1.5)','FontSize',14);
% axes('Position',[0.18,0.55,0.25,0.2]); % magnify the detailed figures
% plot(Iter(4:6),dualres_alm1(4:6),'c-o','LineWidth',2);
% hold on;
% plot(Iter(4:6),dualres_alm2(4:6),'m-x','LineWidth',2);
% axis([4,6,1.25,1.5]);

% plot primal residual
figure(3),
loglog(Iter,primalres_alm1,'c-','LineWidth',2);
hold on;
loglog(Iter,primalres_alm2,'m-','LineWidth',2);
hold on;
loglog(Iter,primalres1,'b-','LineWidth',2);
hold on;
loglog(Iter,primalres2,'r-','LineWidth',2);
hold on;
loglog(Iter,primalres3,'k-','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('||x^{k+1}-z^k||','FontSize',14);
legend('iALM','Prox-iALM','LiMEAL (\eta = 0.5)','LiMEAL (\eta = 1)','LiMEAL (\eta = 1.5)','FontSize',14);
axis([0,NumIter,1e-15,1.15]);


% plot stationary measure
figure(4),
loglog(Iter,staterr1,'b-','LineWidth',2);
hold on;
loglog(Iter,staterr2,'r-','LineWidth',2);
hold on;
loglog(Iter,staterr3,'k-','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('Gradient of Moreau Envelope','FontSize',14);
legend('LiMEAL (\eta = 0.5)','LiMEAL (\eta = 1)','LiMEAL (\eta = 1.5)','FontSize',14);
axis([0,NumIter,1e-15,25]);
% axes('Position',[0.18,0.18,0.38,0.3]); % magnify the detailed figures
% semilogy(Iter(1:1000),staterr1(1:1000),'b-','LineWidth',2);
% hold on;
% semilogy(Iter(1:1000),staterr2(1:1000),'r-','LineWidth',2);
% hold on;
% semilogy(Iter(1:1000),staterr3(1:1000),'k-','LineWidth',2);
% axis([0,1000,1e-14,25]);