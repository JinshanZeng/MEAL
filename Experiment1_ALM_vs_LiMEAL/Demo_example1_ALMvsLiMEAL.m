% Ref: Jinshan Zeng, Wotao Yin and Ding-Xuan Zhou, Moreau envelope 
% augmented Lagrangian method for nonconvex optimization with linear
% constraints, Journal of Scientific Computing, 2021
% Example1: Yu Wang, Wotao Yin and Jinshan Zeng, Global convergence of ADMM in nonconvex nonsmooth optimization,JSC, 2019
% compare the performance of LiMEAL with ALM and ADMM for the following optimization problem 
% minimize_x,y  x^2-y^2    subject to x=y, x\in [-1,1]
clear all; close all; clc;
NumIter = 50;

% run ALM
lamb0 = 0;
beta = 50;
[x,y,lamb,objfun,dualres]=ALM_example1(lamb0,beta,NumIter);

% plot figures
Iter = (1:NumIter)';
figure(1),
plot(Iter,objfun,'r-','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('f(x^k,y^k)','FontSize',14);
title('ALM','FontSize',14);

figure(2),
plot(Iter,dualres,'r--','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('|x^k-y^k|','FontSize',14);
title('ALM','FontSize',14);

figure(3),
plot(Iter,lamb,'r-o','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('\lambda^k','FontSize',14);
% legend('\lambda^k','FontSize',14);
title('ALM','FontSize',14);

% run LiMEAL with different eta's
y0 = 0.5;
x0 = y0;
z0 = [x0;y0];
% beta = 50;
gamma = 0.5;

eta1 = 0.5;
[x1,y1,lamb1,objfun1,dualres1,staterr1]=LiMEAL_example1(x0,y0,z0,lamb0,beta,gamma,eta1,NumIter);

eta2 = 1;
[x2,y2,lamb2,objfun2,dualres2,staterr2]=LiMEAL_example1(x0,y0,z0,lamb0,beta,gamma,eta2,NumIter);

eta3 = 1.5;
[x3,y3,lamb3,objfun3,dualres3,staterr3]=LiMEAL_example1(x0,y0,z0,lamb0,beta,gamma,eta3,NumIter);

% plot figures
Iter = (1:NumIter)';
figure(1),
plot(Iter,objfun,'r-','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('f(x^k,y^k)','FontSize',14);
title('ALM','FontSize',14);

figure(2),
plot(Iter,dualres,'r--','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('|x^k-y^k|','FontSize',14);
title('ALM','FontSize',14);

figure(3),
plot(Iter,lamb,'r-o','LineWidth',2);
xlabel('Iteration','FontSize',14);
legend('\lambda^k','FontSize',14);
title('ALM','FontSize',14);

figure(4),
plot(Iter,objfun1,'m-','LineWidth',2);
hold on;
plot(Iter,objfun2,'k-','LineWidth',2);
hold on;
plot(Iter,objfun3,'b-','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('f(x^k,y^k)','FontSize',14);
legend('\eta=0.5','\eta=1','\eta=1.5','FontSize',14);
title('LiMEAL','FontSize',14);
axes('Position',[0.2,0.35,0.45,0.35]); % magnify the detailed figures
plot(Iter(2:10),objfun1(2:10),'m-','LineWidth',2);
hold on;
plot(Iter(2:10),objfun2(2:10),'k-','LineWidth',2);
hold on;
plot(Iter(2:10),objfun3(2:10),'b-','LineWidth',2);
axis([2,10,-1.5e-3,2.2e-3]);

figure(5),
semilogy(Iter,dualres1,'m-.','LineWidth',2);
hold on;
semilogy(Iter,dualres2,'k--','LineWidth',2);
hold on;
semilogy(Iter,dualres3,'b:','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('|x^k-y^k|','FontSize',14);
axis([0,50,1e-20,1]);
legend('\eta=0.5','\eta=1','\eta=1.5','FontSize',14);
title('LiMEAL','FontSize',14);

figure(6),
plot(Iter,lamb1,'m-.','LineWidth',2);
hold on;
plot(Iter,lamb2,'k--','LineWidth',2);
hold on;
plot(Iter,lamb3,'b:','LineWidth',2);
xlabel('Iteration','FontSize',14);
ylabel('\lambda^k','FontSize',14);
legend('\eta=0.5','\eta=1','\eta=1.5','FontSize',14);
title('LiMEAL','FontSize',14);

figure(7),
semilogy(Iter,staterr1,'m-.','LineWidth',2);
hold on;
semilogy(Iter,staterr2,'k--','LineWidth',2);
hold on;
semilogy(Iter,staterr3,'b:','LineWidth',2);
axis([0,50,1e-20,1]);
xlabel('Iteration','FontSize',14);
ylabel('Gradient of Moreau Envelope','FontSize',14);
legend('\eta=0.5','\eta=1','\eta=1.5','FontSize',14);
title('LiMEAL','FontSize',14);

