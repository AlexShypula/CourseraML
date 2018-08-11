function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% note that the X matrix is assumed to alredy come with the Xo values already with ones
% additionally, the theta vector comes with theta0 in the first index position
% to vectorize the mathematics, we will append a -1 to the theta vector which will be multiplied
% by the y value for the mth training example so it is the sum of
% theta0 x Xo + theta1 x X1 + . . . thetaN x XN + (-1 x y) 

% append the -1 to the theta vector, and call theta_err
% we use a semi-colon, because it is a column

theta_err = [theta ;  -1];

% X_y will be the matrix of X values appended with y values
% so that the product will contain error terms for each m training examples

X_y = [X y];


J = (1/(2*m))*sum((X_y*theta_err).^2) + (lambda/(2*m))*sum(theta(2:end).^2);

% unnormalized grad for each dimension from 0 to n is calculated as 
% the error for the mth observation multiplied by the x value of that dimension
% then all the results are summed up over all m and divided by m
% giving a separate value for each of our dimensions

unnormalized_grad = X'*(X_y*theta_err).*(1/m); 

% to normalize, append zero to  lambda/m multiplied by all other theta vals

norm = [0 ;  (lambda/m)*theta(2:end)];

grad = unnormalized_grad + norm ; 


% =========================================================================

grad = grad(:);

end
