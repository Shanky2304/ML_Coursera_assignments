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

h = sigmoid(X * theta);
val = (y' * log(h)) + ((1 .- y') * log(1 .- h));
J = ((-1/m) * sum(val));

% Here as well we shouldn't consider theta(0) which is theta(1) in octave since
% octave starts from index 1.
reg_term = ((lambda / (2 * m)) * sum(theta(2:end, :) .^ 2));

J = J + reg_term; 



grad(1) = (1/m) * ((h' - y') * (X(:, 1)));

for i = 2 : size(theta)

grad(i) = (1/m) * ((h' - y') * (X(:, i))) + ((lambda * theta(i)) / m)  ;

end


% =============================================================

end
