function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters
% added reg term to avoid overfitting and underfitting. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
%
 h = sigmoid(X*theta);

%thetar = [0;theta(2:end, :);];
%reg = lambda/2 *m + thetar;
%J = sum(-y .* log(h) - (1-y) .* log(1-h)) * (1/m) + reg;
%grad = X' *(h - y) * (1/m) + [0: lambda/m + thetar];h = sigmoid(X*theta);

% reg = lambda/(2 *m)  *(theta(2:end, :)' * (theta(2:end, :));
reg = lambda/(2*m) * (theta(2:end, :)' * theta(2:end,:));
J = sum(-y .* log(h) - (1-y) .* log(1-h)) * (1/m) + reg;
grad = X' *(h - y) * (1/m) + [0; lambda/m * theta(2:end, :)];




% ============================= ================================

end
