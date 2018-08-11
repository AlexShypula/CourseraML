function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Useful values
m = size(X, 1);
n = size(X, 2);
num_labels = size(Theta2, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% calculate the y matrix

% repeat y vector for number of labels
y = repmat(y, 1, num_labels);

%  create logical index of the column values of equal dim as y
[r,c] = find(y);
c=reshape(c, [], num_labels);

% create logical vector for matches
y = c==y; 

% initialize big deltas to iterate over

D_2 = zeros(size(Theta2));
D_1 = zeros(size(Theta1));

% for t=1:10

% multiply by Theta1 (create output from layer 1), outputs a 5000x25 row matrix; take the transpose of theta which is originally 25x401
a_2 = sigmoid(X*Theta1'); 
% append 1's to the a matrix (output layer of all 25 neurons)
a_2 = [ones(m,1) a_2]; 
% multiply a with the bias with Theta2 for the output layer, a 5000x10 matrix from a 5000x26 by 26x10 multiplication
a_3 = sigmoid(a_2*Theta2');

% calculate little delta for layer 3 (output layer)

d_3 = a_3-y; 
d_2=d_3*Theta2.*[ones(m,1) sigmoidGradient(X *Theta1')];

% equal to the following
#  d_2_2=d_3*Theta2.*(a_2.*(1-a_2)); 

% calculate big deltas for layer 2 and layer 3 X is equal to alpha 1
% we need to remove the first row from the d_2, which corresponds to the weights for the 
% bias layer I believe

D_2 = D_2 + d_3'*a_2;
D_1 = D_1 +  d_2(:, 2:end)'*X;


# need to append the ones with length as number of rows in Theta2 ()
Theta2_reg = [zeros(num_labels,1) (lambda/m).* Theta2(:,2:end)];

Theta1_reg = [zeros(hidden_layer_size,1) (lambda/m).*Theta1(:,2:end)];

Theta2_grad = (1/m).*D_2+Theta2_reg;
Theta1_grad = (1/m).*D_1+Theta1_reg;

for i=1:num_labels

J_i=(-1/m)*[y(:,i)'*log(a_3(:,i))+(1-y(:,i))'*log(1-a_3(:,i))];
J=J+J_i; 
endfor

% add regularizaiton to the cost function, lambda/2m mutliplied by the sum of all Theta's squared
% except for the first column of Thetas which happens to be the thetas for the bias
% the (:) turns a matrix into a single unrolled vector

reg = (lambda/(2*m))*(sum([Theta1(:,2:end)(:).^2; Theta2(:,2:end)(:).^2]));

J=J+reg; 

% endfor













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
