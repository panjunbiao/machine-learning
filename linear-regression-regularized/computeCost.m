function J = computeCost(X, y, theta, lambda)
% Compute the cost of fitting X to y with theta, using linear regression algorithms.
% X is a m*n matrix, in which m is the number of samples, and n is the number of features.
% y is a m*1 vector, in which each line of data is the observed value given each of the sample X.
% theta is a n*1 vector, which is the parameters using in linear regression.

m = length(y);

difference = X * theta - y;

thetasize = size(theta, 1);

theta_without_0 = theta(2:thetasize);

penal = lambda * theta_without_0' * theta_without_0;

J = (difference' * difference + penal) / 2 / m;

end