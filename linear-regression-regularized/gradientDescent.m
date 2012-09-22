function [theta, J_history] = gradientDescent(X, y, theta_init, alpha, lambda, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);
theta = theta_init;

for iter = 1 : num_iters

delta = (X' * (X * theta - y) + lambda * [0; theta(2:size(theta,1))]) / m;
theta = theta - alpha * delta;
J_history(iter) = computeCost(X, y, theta, lambda);

end

end
