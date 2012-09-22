function [theta, mu, sigma, J_history] = linearRegression(X, y, alpha, lambda, num_iters)

m = size(X, 1);

mu = mean(X);

sigma = std(X);

X_norm = normalizeFeatures(X, mu, sigma);

X_norm_bias = [ones(m,1) X_norm];

n = size(X_norm_bias, 2);

theta_init = zeros(n, 1);

[theta, J_history] = gradientDescent(X_norm_bias, y, theta_init, alpha, lambda, num_iters);

end