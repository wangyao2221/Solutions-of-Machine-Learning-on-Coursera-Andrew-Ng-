function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
% 求代价函数的正则项也要从第二项开始，也就是theta(2)开始
regularization_item = lambda * (theta(2:end)' * theta(2:end)) / (2 * m);
J = sum(-log(hTheta(X,theta))' * y - log(1 - hTheta(X,theta))' * (1 - y)) / m + regularization_item;

delta = hTheta(X,theta) - y;
grad(1) =  X(:,1)' * delta / m; 
grad(2:end) = (X(:,2:end)' * delta  + lambda * theta(2:end)) / m;
% =============================================================

end
