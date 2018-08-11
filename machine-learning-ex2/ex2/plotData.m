function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

data=load('ex2data1.txt');
y=data(:,3);

x1=data(:,1);
x2=data(:,2);

pos=find(y==1);
neg=find(y==0);

plot(x1(pos), x2(pos), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(x1(neg), x2(neg), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);





% =========================================================================



hold off;

end
