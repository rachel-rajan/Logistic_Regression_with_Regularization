function x_norm = normalize_features(x)
%normalizes the feature vector and returns the normalized feature vector
%Normalize features
m = size(x, 1);
mu = repmat(mean(x),[m,1]);
sigma = repmat(std(x), [m,1]);
x_norm = (x - mu)./sigma;
end

