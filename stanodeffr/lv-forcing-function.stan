functions {
  /* Forward declaration of prey forcing function. */
  real Y(real t, int N_coef, vector coef);
  
  /* Lotka-Volterra ODE equations with predator forcing function.

     See https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations
    
     @param t The single time point at which to evaluate the ODE.
     @param x State variables of populations.
     @param a Prey growth rate parameter (alpha).
     @param b Prey death rate parameter from the presence of predators (beta).
     @return Left-hand side (LHS) of the system of ODE equations (prey only).
   */
  vector lv(real t,
	    vector x,
	    real a,
	    real b,
	    int N_coef,
	    vector coef) {
    vector[1] dx;
    real y_ = Y(t, N_coef, coef);
    dx[1] = a * x[1] - b * x[1] * y_;
    return dx;
  }

  /* Predator forcing function.

     The original predator data is approximated as a continuous,
     differentiable function here using a polynomial because it's
     simple to implement in Stan; ideally one would want to use a more
     precise spline, but that's more difficult.  Implementing a spline
     can be done later once the model is working.

     @param t Single time point at which to evaluate the forcing
              function.
     @return Predator value at time t.
   */
  real Y(real t, int N_coef, vector coef) {
    real predator = 0;
    for (i in 1:N_coef) {
      predator += coef[i] * pow(t, i-1);
    }
    /* Set negative values to zero, otherwise the calculation will fail.*/
    if (predator < 0) {
      return 0;
    }
    return predator;
  }
}

data {
  /* Dimensions. */
  int<lower=1> T;
  int<lower=1> V;

  /* Experimental data. */
  array[T] real ts;
  array[T] vector[V] y;

  /* Coefficients of fitted polynomial. */
  int<lower=1> N_coef;
  vector[N_coef] coef;

  /* Initial conditions. */
  real<lower=0> t0;
  vector<lower=0>[V] y0;
}

parameters {
  /* Prey death rate. */
  real<lower=0> b;
  /* Prey growth rate; must be higher than prey death rate. */
  real<lower=b> a;
}

model {
  array[T] vector[V] mu = ode_rk45(lv, y0, t0, ts, a, b, N_coef, coef);
  // Sample positive priors for all our ODE parameters; the T[0, ]
  // truncates the normal distrubiton at 0 so that negative values are
  // not sampled.
  a ~ normal(0.5, 0.5);
  b ~ normal(0.5, 0.5);
  // The state variables computed by the ODE should be normally
  // distributed.
  for (t in 2:T) {
    y[t] ~ normal(mu[t], 1);
  }
}
