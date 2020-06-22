function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

h = sigmoid(X * theta);
val = (y' * log(h)) + ((1 .- y') * log(1 .- h));
J = ((-1/m) * sum(val));

% Here as well we shouldn't consider theta(0) which is theta(1) in octave since
% octave starts from index 1.
reg_term = ((lambda / (2 * m)) * sum(theta(2:end, :) .^ 2));

J = J + reg_term;


grad(1) = (1/m) * ((h' - y') * (X(:, 1)));


% If you're just reading the code it's somewhat difficult to see why we had to take 
% transpose of the regularized term but rest of the operations produce a 1xn matrix
% and the rg term was initially nx1, to make addition possible we had to take a transpose
% if you write down the results of each operation you'll see it what we actually want too.

grad(2:end) = (1/m) .* ((h' - y') * (X(:, 2:end))) + (((lambda .* theta(2:end)) ./ m))';







% =============================================================

grad = grad(:);

end
