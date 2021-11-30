% LiMEAL: 
% Example1 from: Yu Wang, Wotao Yin and Jinshan Zeng, Global convergence of ADMM
% in nonconvex nonsmooth optimization,JSC, 2019
% compare the performance of LiMEAL with ALM and ADMM for the following optimization problem
% minimize_x,y  x^2-y^2    subject to x=y, x\in [-1,1]
% By Theorem 3 in (Zeng-Yin-Zhou'2021), LiMEAL converges in this case with
% some appropriate parameters

function [xx,yy,lambda,objfun,dualres,staterr]=LiMEAL_example1(x0,y0,z0,lamb0,beta,gamma,eta,NumIter)
% Input:
% lamb0 -- initialization of Lagrangian multipliers
% x0,y0,z0 -- initialization
% gamma -- proximal parameter (0,1/2) in this case
% eta -- step size (0,2), eta=1 then LiMEAL reduces to proximal ALM
% beta -- penalty parameter (beta>4)
% NumIter -- number of iterations

% Output:
% x,y,lamb -- the outputs of algorithmic iterative sequences
% objfun -- the trend of objective function
% dualres -- dual residual defined as: dualres = ||Ax-b||
% staterr -- stationary error defined as the norm of gradient of Moreau Envelope
objfun = zeros(NumIter,1);
dualres = zeros(NumIter,1);
staterr = zeros(NumIter,1);
xx = zeros(NumIter,1);
yy = zeros(NumIter,1);
lambda = zeros(NumIter,1);

D = 16 - 2*beta*gamma^(-1) - gamma^(-2);
for i=1:NumIter
    xi1 = 2*x0 + gamma^(-1)*z0(1) - lamb0;
    xi2 = 2*y0 - gamma^(-1)*z0(2) - lamb0;
    Dx = (4-beta-gamma^(-1))*xi1 + beta*xi2;
    Dy = -beta*xi1+(4+beta+gamma^(-1))*xi2;
    tempx = Dx/D;
    x = (abs(tempx)>1)*sign(tempx) + (1-(abs(tempx)>1))*tempx; % projection onto [-1,1]
    y = Dy/D;
    xy = [x;y];
    z = z0 - eta*(z0-xy);
    lamb = lamb0 + beta*(x-y);
    xx(i)=x;
    yy(i)=y;   
    lambda(i)=lamb;
    objfun(i)=x^2-y^2;
    dualres(i)= norm(x-y);
    xy0 = [x0;y0];
    staterr(i)= sqrt(norm(gamma^(-1)*(z0-xy)+(xy-xy0))^2+beta^(-2)*norm(lamb-lamb0)^2);
    % print the main evaluation metrics
    fprintf('NumIter: %f, objfun: %g, dualres: %g, staterr: %g, x:%g, y:%g, lambda: %g \n',i,objfun(i),dualres(i),staterr(i),x,y,lamb);
    
    x0 = x;
    y0 = y;
    z0 = z;
    lamb0 = lamb;
end
clear xi1; clear xi2; clear D; clear Dx; clear Dy; clear xy; clear tempx; clear xy0;
end