/*ZcurveEM_PL
* author: Paweł Lenartowicz
* email:  pawellenartowicz@europe.com
*
*/
#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

inline NumericVector c_dnorm(const NumericVector& x, double mu, double sigma) {
  constexpr double inv_sqrt_2pi = 0.3989422804014327;
  NumericVector z = (x - mu) / sigma;
  return inv_sqrt_2pi / sigma * exp(-0.5 * pow(z, 2.0));
}

inline double c_pnorm(double x, double mu, double sigma) {
  double z = (x - mu) / sigma;
  return 0.5 * (1.0 + std::erf(z / 1.4142135623730951));
}

NumericMatrix calculate_log_likelihood(const NumericVector& x, const NumericVector& mu, const NumericVector& sigma, double a, double b) {
  NumericMatrix ll(x.size(), mu.size());
  NumericVector l1(x.size()), l2(x.size()); // Preallocate

  for (int k = 0; k < mu.size(); ++k) {
    double denom = c_pnorm(b, mu[k], sigma[k]) - c_pnorm(a, mu[k], sigma[k])
                 + c_pnorm(-a, mu[k], sigma[k]) - c_pnorm(-b, mu[k], sigma[k]);

    l1 = c_dnorm(x, mu[k], sigma[k]);
    l2 = c_dnorm(-x, mu[k], sigma[k]);

    ll(_, k) = log(l1 + l2) - log(denom); // Use cached denominator
  }
  return ll;
}

void filter_z_values(NumericVector& x, double a, double b) {
  x = x[x > a & x < b]; // Filter in-place
}

NumericMatrix e_step_PL(NumericMatrix u_log_lik, NumericVector log_theta, NumericVector& l_row_sum, NumericVector& log_l_row_sum) {
  int n = u_log_lik.nrow();
  int m = u_log_lik.ncol();

  NumericMatrix probabilities(n, m);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < m; ++j) {
      probabilities(i, j) = exp(u_log_lik(i, j) + log_theta[j]);
      row_sum += probabilities(i, j);
    }

    l_row_sum[i] = row_sum;
    log_l_row_sum[i] = log(row_sum); // Cache log(row_sum)

    for (int j = 0; j < m; ++j) {
      probabilities(i, j) /= row_sum;
    }
  }

  return probabilities;
}

NumericVector m_step_PL(NumericMatrix p){
  NumericVector new_theta (p.ncol());

  for(int k = 0; k < p.ncol(); k++){
    new_theta[k] = sum(p(_,k)) / p.nrow();
  }
  return(new_theta);
}

double get_prop_high_PL(NumericVector x, double select_sig, double b){
  double a = c_pnorm(select_sig/2, 0, 1);

  LogicalVector x_sig_true = x > a;
  NumericVector x_sig = x[x_sig_true];

  LogicalVector x_high_true = x > b;
  NumericVector x_high = x[x_high_true];

  double prop_high = (1.0 * x_high.length()) / (1.0 * x_sig.length());
  return prop_high;
}

// [[Rcpp::export(.zcurve_EM_PL_RCpp)]]
List zcurve_EM_PL_RCpp(NumericVector x, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level,
                 int max_iter, double criterion) {

  double prop_high = get_prop_high_PL(x, sig_level, b);
  filter_z_values(x, a, b);

  NumericMatrix u_log_lik = calculate_log_likelihood(x, mu, sigma, a, b);
  NumericVector log_theta = log(theta);
  NumericVector l_row_sum(x.size(), 0.0); // Preallocate
  NumericVector log_l_row_sum(x.size(), 0.0); // Preallocate
  NumericMatrix p(x.size(), mu.size());
  NumericVector Q(max_iter + 1, 0.0);

  int i = 0;

  do {
    // Combined E-step
    p = e_step_PL(u_log_lik, log_theta, l_row_sum, log_l_row_sum);

    // M-step
    theta = m_step_PL(p);
    log_theta = log(theta); // Update log(theta)

    Q[i + 1] = sum(log_l_row_sum); // Use cached log(row_sum)
    ++i;

  } while ((fabs(Q[i] - Q[i - 1]) >= criterion) && (i < max_iter));

  List ret;
  ret["iter"] = i;
  ret["Q"] = Q[i];
  ret["mu"] = mu;
  ret["weights"] = theta;
  ret["sigma"] = sigma;
  ret["prop_high"] = prop_high;

  return ret;
}
