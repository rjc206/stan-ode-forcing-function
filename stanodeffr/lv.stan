functions{
  vector lv(real t,
	    vector y,
	    real a, real b, real g, real d){
    vector[2] dydt;
    //y[1] and y[2] correspond to x and y respectively.
    dydt[1] =  a*y[1] - b*y[1]*y[2];
    dydt[2] = -g*y[2] + d*y[1]*y[2];
    return dydt;
  }
}

data {
  int<lower=1> T;
  array[T] vector[2] y;
  vector<lower=0>[2] y0;
  real<lower=0> t0;
  array[T] real ts;
  real<lower=0> g;
  real<lower=0> d;
}

parameters {
  // y0 contains the initial values of x and y respectively.
  real<lower=0> a;
  real<lower=0> b;
}

model {
  array[T] vector[2] mu = ode_rk45(lv, y0, t0, ts, a, b, g, d);
  // Sample positive priors for all our ODE parameters.
  a ~ normal(0.5, 0.5);
  b ~ normal(0.5, 0.5);
  // The state variables computed by the ODE should be lognormally
  // distributed.
  for (t in 1:T) {
    y[t] ~ normal(mu[t], 1);
  }
}

