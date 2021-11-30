% Proximal inexact ALM (Zhang&Luo, SIOPT 2020) for quadratic programming
% Considering the following quadratic programming:
% min_x 1/2 x^TQx+r^Tx  s.t. Ax=b, ell_i<=xi<=ui for all i
% Ref: Jiawei Zhang and Zhi-Quan Luo, A proximal alternating direction
% method of multiplier for linearly constrained nonconvex minimization,
% SIAM J. Optim., 30(3): 2272-2302, 2020.
function [x,z,lamb,objfun,dualres,primalres]=ProxiALM_QP(QP,x0,z0,lamb0,p,beta,alpha,s,eta,NumIter)
% Input:
% QP: the settings of the problem of quadratic programming
% QP.Q: matrix Q; QP.r: vector r; QP.A: matrix A; QP.b: vector b;
% QP.l: lower bound ell; QP.u: upper bound u
% lamb0 -- initialization of Lagrangian multipliers
% x0,z0 -- initialization
% p: inverse of proximal parameter (p>rho, where rho is the weakly convex modulu,equaling to ||Q|| in this case)
% beta: penalty parameter in the augmented Lagrangian
% s -- primal stepsize (0,1/(||Q||+p+beta*||A||^2)), where p>rho, Gamma: penalty para.
% eta -- z-step size (0,1], eta=1 then ProxiALM reduces to iALM
% alpha -- dual stepsize
% NumIter -- number of iterations

% Output:
% x,z,lamb -- the outputs of algorithmic iterative sequences
% objfun -- the trend of objective function
% dualres -- dual residual defined as: dualres = ||Ax-b||
% primalres -- primal residual defined as: primalres = ||x(k+1)-z(k)||

objfun = zeros(NumIter,1);
dualres = zeros(NumIter,1);
primalres = zeros(NumIter,1);

% calculating the vector beta*A'b-r
Abr = beta*(QP.A)'*(QP.b)-QP.r;
% run iterations
for i=1:NumIter
    gradx = QP.Q*x0+(QP.A)'*(beta*QP.A*x0+lamb0)+p*(x0-z0)-Abr;
    tempx = x0 - s*gradx;
    x = (tempx>=QP.u).*QP.u+(tempx<=QP.ell).*QP.ell+(QP.ell<tempx&tempx<QP.u).*tempx;% projection onto [ell,u]
    z = z0-eta*(z0-x);
    lamb = lamb0+alpha*(QP.A*x-QP.b);
    
    objfun(i)=1/2*x'*QP.Q*x+(QP.r)'*x; % objective function
    dualres(i)= norm(QP.A*x-QP.b); % dual residual
    primalres(i) = norm(x-z0); % primal residual
    % print the main evaluation metrics
    fprintf('NumIter: %g, objfun: %g, dualres: %g, primalres: %g \n',i,objfun(i),dualres(i),primalres(i));
    
    x0 = x;
    z0 = z;
    lamb0 = lamb;
end
clear Abr; clear gradx; clear tempx;
end