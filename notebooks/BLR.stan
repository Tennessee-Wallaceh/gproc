BLR_model = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  array[n] int<lower=0,upper=1> y; // outputs
  matrix[n, d] x; // inputs
}
parameters {
  vector[d] theta; // regression parameter
}
model {
  theta ~ normal(0, 1); // prior on theta
  vector[n] p = x * theta; 
  y ~ bernoulli_logit(p) // target
}
"""