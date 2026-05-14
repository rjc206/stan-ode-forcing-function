To build the vignettes in R:

1. Install the dependencies:

   ```r
   if (! require("remotes", quietly = TRUE)) install.packages("remotes")
   remotes::install_deps(dependencies = TRUE)
   ```

1. Build the specific vignette to see the output HTML and PDF files, for
   example:

   ```r
   rmarkdown::render("01-stan-ode-forcing-function.Rmd", output_format = "all")
   ```
