FROM rocker/verse:4.4.1
RUN apt-get update && apt-get install -y --no-install-recommends libv8-dev && rm -rf /var/lib/apt/lists/*
RUN R -q -e 'install.packages(c("rstan", "rstantools", "bayesplot"), repos = "https://cloud.r-project.org")'
