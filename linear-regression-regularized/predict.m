function y = predict(X, mu, sigma, theta)
m = size(X, 1);
X_norm = normalizeFeatures(X, mu, sigma);
X_norm_bias = [ones(m,1) X_norm];
y = X_norm_bias * theta;
end
