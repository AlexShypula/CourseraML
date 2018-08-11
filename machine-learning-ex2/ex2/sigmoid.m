function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
# we don't want inverse, because we want to apply the sigmoid function to each
# element of the matrix 

g=(1.+e.^(-z)).^(-1);


% =============================================================

end
