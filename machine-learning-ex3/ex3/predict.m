function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

# append ones
X = [ones(m,1) X];
# multiply by Theta1 (create output from layer 1), outputs a 5000x25 row matrix; take the transpose of theta which is originally 25x401
a = sigmoid(X*Theta1'); 
# append 1's to the a matrix (output layer of all 25 neurons)
a = [ones(m,1) a]; 
# multiply a with the bias with Theta2 for the output layer, a 5000x10 matrix from a 5000x26 by 26x10 multiplication
b = sigmoid(a*Theta2'); 
[max_pred,p] = max(b, [], 2);








% =========================================================================


end
