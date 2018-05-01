#include <RcppArmadillo.h>
using namespace Rcpp;

// Light version of L2 norm - Euclidean dist
// [[Rcpp::depends(RcppArmadillo)]]
arma::mat lite_l2(const arma::mat& X, const arma::mat& Y) {

  int N = X.n_rows;
  int K = Y.n_rows;

  arma::mat x_dot = sum(pow(X, 2), 1);
  arma::mat y_dot = trans(sum(pow(Y, 2), 1));
  arma::mat xy = (2 * X) * trans(Y);
  arma::mat D = x_dot * arma::ones(1, K);
  D = D + arma::ones(N, 1) * y_dot;
  D = sqrt(D - xy);

  return (D);
}

// Lloyd's Algorithm
// [[Rcpp::export]]
List kmeans_cpp(const arma::mat& X, int k, int max_iter) {

  // initialize scalars
  int n = X.n_rows; // observations
  int d = X.n_cols; // features
  int iter_ct = 0; // counter
  int delta_ct = 100000; // large number to optimality check
  int is_opt = 0; // optimality indicator (1 - fail, 0 - pass)

  // intialize matrix of centroid features
  arma::mat k_mat = arma::zeros(k, d);

  // intialize randomly selected centers by row index
  arma::ivec k_index_vec = arma::randi( k, arma::distr_param(0,(n-1)) );
  // intialize matrix for new centers
  for (int i = 0; i < k ; i ++) {
    k_mat.row(i) = X.row(k_index_vec(i));
  }

  // Preserve cluster initialization
  arma::mat km_start_ctr = arma::zeros(k, d);
  km_start_ctr = k_mat;

  // intialize placeholder matrix for previous best
  arma::mat k_mat_old = arma::zeros(k, d);
  arma::mat dist_to_k = arma::zeros(n, k);
  arma::umat min_index(n, 1);
  arma::umat min_index_old(n, 1);
  min_index_old.fill(-1);

  // begin iterating
  while ( iter_ct < max_iter && is_opt == 0 ) {

    // compute dist and classifications - by index
    arma::mat dist_to_k = lite_l2(X, k_mat);
    // argmin step
    min_index = index_min(dist_to_k, 1); // dim=1 corresponds to min row index

    // check for changes in assignment and update optimality
    arma::umat delta = abs(min_index - min_index_old);
    delta_ct = as_scalar(sum(delta));
    if ( delta_ct == 0 ) {
      is_opt = 1;
    } else {
      min_index_old = min_index;
    }

    // update reference centers
    k_mat_old = k_mat;

    // recompute centers as the average of conponents
    for (int ii = 0; ii < k ; ii ++) {

      // identify minimal distance subset for center k_i
      arma::uvec is_k_ii = find(min_index == ii);
      int n_obj = is_k_ii.n_elem;
      arma::mat X_sub_ii(n_obj, d);
      X_sub_ii = X.rows(is_k_ii);

      // update center k_ii as average of all classified
      k_mat.row(ii) = mean(X_sub_ii, 0); // column mean 0
    }

    // increment count
    iter_ct = iter_ct + 1;
  }

  return List::create(Named("n_change") = delta_ct,
                      Named("n_iter") = iter_ct,
                      Named("classification") = min_index,
                      Named("start_centers") = km_start_ctr,
                      Named("centers") = k_mat);
}
