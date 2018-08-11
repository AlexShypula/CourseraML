function [theta0_vals, theta1_vals] = gradientDescentTheta(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

%data=load('ex1data1.txt');
%y=data(:,2);
m = length(y); % number of training examples
%X=[ones(m,1),data(:,1)];


%theta=zeros(2,1);


%alpha=0.01;
%num_iters=1500;


J_history = zeros(num_iters, 1);
theta0_vals = zeros(num_iters, 1);
theta1_vals = zeros(num_iters, 1);
for iter = 1:num_iters, 

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


foo_1=theta(1)-(alpha/m)*(sum((theta'*X')'-y));
foo_2=theta(2)-(alpha/m)*(sum(((theta'*X')'-y).*X(:,2)));

theta(1)=foo_1;
theta(2)=foo_2;


    % ============================================================

    % Save the cost J in every iteration    


J_history(iter)=sum(((theta'*X')'-y).^2)/(2*m);
theta0_vals(iter) = theta(1);
theta1_vals(iter) = theta(2);

%predict1=[1,3.5]*theta;
%predict2=[1,7]*theta;

end

