library(devtools)
# library(Rtools)
library(Rcpp)
library(RcppArmadillo)
library(data.table)
library(ggplot2)
library(foreach)

find_rtools()
has_devel()

## Code path
c.path <- ""

## Compile C++ functions
Rcpp::sourceCpp(paste0(c.path, "arma_kmeans.cpp"))
Rcpp::sourceCpp(paste0(c.path, "arma_kmeans_plusplus.cpp"))

## Generate 3 cluster data set w/ ambiguous middle set
set.seed(123)
three_clst <- matrix(c(rnorm(25, 3, 1), rnorm(25, -3, 1)
                       , rnorm(25, -.25, 1), rnorm(25, 4, 1)
                       , rnorm(25, -2, 1), rnorm(25, 0.25, 1))
                     , 75, 2, byrow = F)

## Test efficiency - neglible for toy data
system.time(kmeans_cpp(X = three_clst, k = 3, max_iter = 50))
system.time(kmeans(three_clst, centers = 3, algorithm = "Lloyd", iter.max = 15))

set.seed(1234)
k_sims <- foreach(ii = 1:10000, .combine = "rbind", .errorhandling = "remove") %do% {
  kmpp_iter <- kmeanspp_cpp(X = three_clst, k = 3, max_iter = 25)$n_iter
  km_iter <- kmeans_cpp(X = three_clst, k = 3, max_iter = 25)$n_iter
  return(data.table(sim_i = ii, km_iter, kmpp_iter))
}

ggplot(data = k_sims, aes(x = km_iter)) + geom_histogram(fill = "steelblue", alpha = 0.5, bins = 15) +
  geom_histogram(aes(x = kmpp_iter), fill = "green3", alpha = 0.5, bins = 15)

## kmeans++
kmpp_test3 <- kmeanspp_cpp(X = three_clst, k = 3, max_iter = 25)
## kmeans
km_test3 <- kmeans_cpp(X = three_clst, k = 3, max_iter = 25)

## visualizing single run
par(mfrow = c(1,2))
plot(three_clst, type = "p", main = paste0("N steps: ", kmpp_test3$n_iter))
points(kmpp_test3$start_centers, col = "red", pch = 16)
points(kmpp_test3$centers, col = "green", pch = 16)

plot(three_clst, type = "p", main = paste0("N steps: ", km_test3$n_iter))
points(km_test3$start_centers, col = "blue", pch = 16)
points(km_test3$centers, col = "orange", pch = 16)
par(mfrow = c(1,1))
