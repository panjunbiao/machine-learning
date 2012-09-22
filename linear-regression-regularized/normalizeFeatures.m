function X_norm = normalizeFeatures(X, mu, sigma)

fprintf('There is two warnings here.\n');
X_norm = zeros(size(X,1), size(X,2));
X_norm = (X .- mu) ./ sigma;
fprintf('There is two warnings above.\n');

end