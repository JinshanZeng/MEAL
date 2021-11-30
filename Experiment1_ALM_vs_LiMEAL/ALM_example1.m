% Augmented Lagrangian method
% Example1: Yu Wang, Wotao Yin and Jinshan Zeng, Global convergence of ADMM
% in nonconvex nonsmooth optimization,JSC, 2019
% compare the performance of LiMEAL with ALM and ADMM for the following optimization problem
% minimize_x,y  x^2-y^2    subject to x=y, x\in [-1,1]
% By Proposition 1 in Wang-Yin-Zeng'2019, ALM with bounded beta diverges for this example.

function [xx,yy,lambda,objfun,dualres]=ALM_example1(lamb0,beta,NumIter)
% Input:
% lamb0 -- initialization of Lagrangian multipliers
% beta -- penalty parameter (>2)
% NumIter -- number of iterations

% Output:
% xx,yy,lambda -- the outputs of algorithmic iterative sequences
% objfun -- the trend of objective function
% dualres -- dual residual defined as: dualres = ||Ax-b||
objfun = zeros(NumIter,1);
dualres = zeros(NumIter,1);
xx = zeros(NumIter,1);
yy = zeros(NumIter,1);
lambda = zeros(NumIter,1);
for i=1:NumIter
    x = sign(lamb0) + (lamb0==0)*(2*(rand(1)>0.5)-1);
    y = x+(lamb0+2*x)/(beta-2);
    lamb = lamb0 + beta*(x-y);
    xx(i)=x;
    yy(i)=y;
    lambda(i)=lamb;
    objfun(i)=x^2-y^2;
    dualres(i)= norm(x-y);
    % print the main evaluation metrics
    fprintf('NumIter: %f, objfun: %g, dualres: %g, x: %g, y:%g, lambda: %g \n',i,objfun(i),dualres(i),x, y, lamb);
    
    lamb0=lamb;
end
clear templamb0;
end