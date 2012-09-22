% This is the main procedure which performs linear regression with gradient descent.
% This algorithm support multiple variables linear regression, enabling multiple features.
% This algorithm is my study notes in the Machine Learning course by Andrew Ng.
% See https://class.coursera.org/ml-2012-002/class/index for more details about this algorithm.

clear;
close all;
clc;

fprintf('Start regularized linear regression...\n');

% Load data for training from train_data.txt
data = load('train_data.txt');

% Get the number of colums of samples.
data_n = size(data, 2);

% Treat the first data_n - 1 columns as X
X = data(:, 1:(data_n - 1));

% Treat the last column as y
y = data(:, data_n);

% Set the learning rate alpha, which can be adjusted in different learning cases.
alpha = 0.01;

% Set the penalty parameter lambda to minimize the dependence on theta.
lambda = 0;

% Set the number of iterations, which can be adjusted in defferent learning cases.
num_iters = 500;

% Call the linear regression function to get trained theta, mean of X,
% stand deviation of X, and the cost of each iteration.
[theta, mu, sigma, J_history] = linearRegression(X, y, alpha, lambda, num_iters);

figure(1);
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

X_predict = load('predict_data.txt');
p = predict(X_predict, mu, sigma, theta);
fprintf('Predicted values given input X:\n');
[X_predict p]
