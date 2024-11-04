data {
  int<lower=0> n_obs;
  int<lower=1> data_dim;
  array[n_obs] vector[data_dim] x;
}

parameters {
  vector[data_dim] theta;
}

model {
  vector[data_dim] mu1 = theta;
  vector[data_dim] mu2 = -1.0 * theta;
  matrix[data_dim, data_dim] Sigma1 = rep_matrix(0, data_dim, data_dim);
  matrix[data_dim, data_dim] Sigma2 = rep_matrix(0, data_dim, data_dim);

  for (d in 1:data_dim) {
    Sigma1[d, d] = 0.5;
    Sigma2[d, d] = 0.5;
  }

  for (n in 1:n_obs) {
    target += log_mix(0.5,
                      multi_normal_lpdf(x[n] | mu1, Sigma1),
                      multi_normal_lpdf(x[n] | mu2, Sigma2));
  }

  target += normal_lpdf(theta[1] | 0, 1);
  target += normal_lpdf(theta[2] | 0, 1);
}
