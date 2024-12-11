functions {
  // Function returns a posterior predictive sample given hyperparameter values
  vector gp_pred_rng(array[] real x_star, array[] real x, vector y, vector y_se,
                     real sigma_SE, real sigma_M32, real sigma_P,
                     real ell_SE, real ell_M32, real ell_P,
                     real T,
                     real jitter) {
    int N = rows(y);
    int N_star = size(x_star);
    vector[N_star] f_star;
    {
      matrix[N, N] K;
      matrix[N, N] L;
      vector[N] alpha;
      matrix[N, N_star] k_x_xstar;
      matrix[N, N_star] v;
      vector[N_star] fstar_mu;
      matrix[N_star, N_star] fstar_cov;

      K = diag_matrix(square(y_se));
      k_x_xstar = rep_matrix(0, N, N_star);
      fstar_cov = rep_matrix(0, N_star, N_star);

      // Include the respective kernel term iff both hyperparameters are positive
      if (sigma_SE > 0 && ell_SE > 0) {
        K = K + gp_exp_quad_cov(x, sigma_SE,  ell_SE);
        k_x_xstar = k_x_xstar + gp_exp_quad_cov(x, x_star, sigma_SE, ell_SE);
        fstar_cov = fstar_cov + gp_exp_quad_cov(x_star, sigma_SE, ell_SE);
      }
      if (sigma_M32 > 0 && ell_M32 > 0) {
        K = K + gp_matern32_cov(x, sigma_M32 , ell_M32);
        k_x_xstar = k_x_xstar + gp_matern32_cov(x, x_star, sigma_M32, ell_M32);
        fstar_cov = fstar_cov + gp_matern32_cov(x_star, sigma_M32, ell_M32);
      }
      if (sigma_P > 0 && ell_P > 0 && T > 0) {
        K = K + gp_periodic_cov(x, sigma_P,  ell_P, T);
        k_x_xstar = k_x_xstar + gp_periodic_cov(x, x_star, sigma_P, ell_P, T);
        fstar_cov = fstar_cov + gp_periodic_cov(x_star, sigma_P, ell_P, T);
      }

      L = cholesky_decompose(K);
      alpha = mdivide_left_tri_low(L, y);
      alpha = mdivide_right_tri_low(alpha', L)';

      fstar_mu = k_x_xstar' * alpha;

      v = mdivide_left_tri_low(L, k_x_xstar);
      fstar_cov = fstar_cov - v' * v;

      f_star = multi_normal_rng(fstar_mu, add_diag(fstar_cov, jitter));
    }
    return f_star;
  }
}

data {
  int<lower=1> N;
  array[N] real x;
  vector[N] y; // observed flux density
  vector[N] y_stderr; // standard error of observed flux density
  int<lower=1> N_star;
  array[N_star] real x_star;
  real x_mingap;
  real x_range;
  real T_lower;
  real T_upper;
}
transformed data {
  vector[N] mu = rep_vector(0, N); // Zero mean function
  real delta = 1e-9; // jitter to ensure pos. semi-def. of covariance matrix
}
parameters {
  real<lower=0> sigma_SE;
  real<lower=0> sigma_M32;
  real<lower=0> sigma_P;
  real<lower=x_mingap> ell_M32;
  real<lower=ell_M32> ell_SE;
  real<lower=x_mingap> ell_P;
  real<lower=T_lower,upper=T_upper> T;
}
model {
  matrix[N, N] K = gp_exp_quad_cov(x, sigma_SE, ell_SE) +
                   gp_matern32_cov(x, sigma_M32, ell_M32) +
                   gp_periodic_cov(x, sigma_P, ell_P, T);

  K = add_diag(K, delta);
  K = add_diag(K, square(y_stderr));

  matrix[N, N] L = cholesky_decompose(K);

  ell_SE ~ inv_gamma(3, 0.5*x_range);
  ell_M32 ~ inv_gamma(3, 0.5*x_range);
  ell_P ~ inv_gamma(3, 0.5*x_range);
  sigma_SE ~ std_normal();
  sigma_M32 ~ std_normal();
  sigma_P ~ std_normal();
  // stan automatically assigns uniform prior to T

  y ~ multi_normal_cholesky(mu, L);
}
generated quantities {
  vector[N_star] f_star;
  vector[N_star] f_star_SE;
  vector[N_star] f_star_M32;
  vector[N_star] f_star_P;
  vector[N] f_prior;

  // Prior predictive samples
  real<lower=0> sigma_SE_sim = abs(normal_rng(0,1));
  real<lower=0> sigma_M32_sim = abs(normal_rng(0,1));
  real<lower=0> sigma_P_sim = abs(normal_rng(0,1));
  real ell_M32_sim = inv_gamma_rng(3, 0.5*x_range);
  real ell_SE_sim = inv_gamma_rng(3, 0.5*x_range);
  real ell_P_sim = inv_gamma_rng(3, 0.5*x_range);
  real T_sim = uniform_rng(T_lower, ell_P_sim);
  matrix[N, N] K_sim;

  K_sim = gp_exp_quad_cov(x, sigma_SE_sim,  ell_SE_sim) +
          gp_matern32_cov(x, sigma_M32_sim, ell_M32_sim) +
          gp_periodic_cov(x, sigma_P_sim,  ell_P_sim, T_sim);

  f_prior = multi_normal_rng(mu, K_sim);

  // Posterior predictive samples
  f_star = gp_pred_rng(x_star, x, y, y_stderr,
                       sigma_SE, sigma_M32, sigma_P,
                       ell_SE, ell_M32, ell_P,
                       T,
                       delta);

  // Posterior predictive samples from each component kernel term
  f_star_SE = gp_pred_rng(x_star, x, y, y_stderr, sigma_SE, 0, 0, ell_SE, 0, 0, 0, delta);
  f_star_M32 = gp_pred_rng(x_star, x, y, y_stderr, 0, sigma_M32, 0, 0, ell_M32, 0, 0, delta);
  f_star_P = gp_pred_rng(x_star, x, y, y_stderr, 0, 0, sigma_P, 0, 0, ell_P, T, delta);
}
