function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.


% data=load('ex2data1.txt');
% y=data(:,3);

% Initialize some useful values
m = length(y); % number of training examples

% x1=data(:,1);
% x2=data(:,2);

% pos=find(y==1);
% neg=find(y==0);

% plot(x1(pos), x2(pos), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
% plot(x1(neg), x2(neg), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% theta=zeros(3,1);
% X=[ones(m,1),data(:,1:2)];


% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

J=(1/m)*sum(-y'*log(sigmoid(theta'*X'))'-(1-y)'*log(1-sigmoid(theta'*X'))');
grad=X'*(sigmoid(theta'*X')'-y)*(1/m);





% =============================================================

end
