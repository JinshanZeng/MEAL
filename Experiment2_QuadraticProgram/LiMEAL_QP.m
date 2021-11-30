% LiMEAL for quadratic programming
% Considering the following quadratic programming:
% min_x 1/2 x^TQx+r^Tx  s.t. Ax=b, ell_i<=xi<=ui for all i
% Ref: Jinshan Zeng, Wotao Yin and Ding-Xuan Zhou, Moreau envelope
% augmented Lagrangian method for noncovnex optimization with linear
% constraints, JSC, 2021.
function [x,z,lamb,objfun,dualres,primalres,staterr]=LiMEAL_QP(QP,x0,z0,lamb0,beta,gamma,eta,NumIter)
% Input:
% QP: the settings of the problem of quadratic programming
% QP.Q: matrix Q; QP.r: vector r; QP.A: matrix A; QP.b: vector b;
% QP.l: lower bound ell; QP.u: upper bound u
% lamb0 -- initialization of Lagrangian multipliers
% x0,z0 -- initialization
% gamma -- proximal parameter (0,1/||Q||) in this case
% eta -- z-step size (0,2), eta=1 then LiMEAL reduces to proximal ALM
% beta -- penalty parameter (sufficiently large), used as dual stepsize
% NumIter -- number of iterations

% Output:
% x,z,lamb -- the outputs of algorithmic iterative sequences
% objfun -- the trend of objective function
% dualres -- dual residual defined as: dualres = ||Ax-b||
% primalres -- primal residual defined as: primalres = ||x(k+1)-z(k)||
% staterr -- stationary error defined as the norm of gradient of Moreau Envelope
objfun = zeros(NumIter,1);
dualres = zeros(NumIter,1);
primalres = zeros(NumIter,1);
staterr = zeros(NumIter,1);

% calculating an inverse of matrix beta*A'A+1/gamma*I
% invAA = (beta*(QP.A)'*QP.A+eye(size(QP.A,2))/gamma)\eye(size(QP.A,2));
invAA = inv(beta*(QP.A)'*QP.A+eye(size(QP.A,2))/gamma);

% calculating the vector beta*A'b-r
Abr = beta*(QP.A)'*(QP.b)-QP.r;
% run iterations
for i=1:NumIter
    tempx = invAA*(z0/gamma-QP.Q*x0-(QP.A)'*lamb0+Abr);
    x = (tempx>=QP.u).*QP.u+(tempx<=QP.ell).*QP.ell+(QP.ell<tempx&tempx<QP.u).*tempx;% projection onto [ell,u]
    z = z0-eta*(z0-x);
    lamb = lamb0+beta*(QP.A*x-QP.b);
    
    objfun(i)=1/2*x'*QP.Q*x+(QP.r)'*x; % objective function
    dualres(i)= norm(QP.A*x-QP.b); % dual residual
    primalres(i) = norm(x-z0); % primal residual
    staterr(i)= sqrt(norm(gamma^(-1)*(z0-x)+QP.Q*(x-x0))^2+beta^(-2)*norm(lamb-lamb0)^2);
    % print the main evaluation metrics
    fprintf('NumIter: %g, objfun: %g, dualres: %g, primalres: %g, staterr: %g\n',i,objfun(i),dualres(i),primalres(i),staterr(i));
    
    x0 = x;
    z0 = z;
    lamb0 = lamb;
end
clear invAA; clear Abr; clear tempx;
end