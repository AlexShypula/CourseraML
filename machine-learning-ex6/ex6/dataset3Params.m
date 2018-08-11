function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];


model_matrix = zeros(length(C_vec)*length(sigma_vec), 3); 

loop_index = 1;
for i = 1:length(C_vec)
    C = C_vec(i); 
    for j = 1:length(sigma_vec)
        sigma = sigma_vec(j);
        % train model
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval); 
        err = mean(double(predictions ~= yval)); 
        % append the results to the matrix
        model_matrix(loop_index, :) = [C,sigma, err]; 
        fprintf("C is");
        disp(C);
        fprintf("sigma is");
        disp(sigma);
        fprintf("error is");
        disp(err);
        % then add one to the loop_index variable to set the right index for next iteration
        loop_index = loop_index + 1; 
     endfor
endfor

model_matrix = sortrows(model_matrix, 3);

C = model_matrix(1,1);
sigma = model_matrix(1,2);


% =========================================================================

end
