data {
  int<lower=0> N;
  vector[N] cost;
  vector[N] success;
  vector[N] emotion;
  int<lower=0,upper=1> y[N];
}

parameters {
  real beta_0;
  real beta_cost;
  real beta_success;
  real beta_emotion;
}

model {
  // Priors
  beta_0 ~ normal(0, 5);
  beta_cost ~ normal(0, 5);
  beta_success ~ normal(0, 5);
  beta_emotion ~ normal(0, 5);

  // Likelihood
  y ~ bernoulli_logit(beta_0 
        + beta_cost * cost
        + beta_success * success
        + beta_emotion * emotion);
}
