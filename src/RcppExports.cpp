// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// zdist_lpdf
NumericVector zdist_lpdf(NumericVector x, double mu, double sigma, double a, double b);
RcppExport SEXP _zcurve_zdist_lpdf(SEXP xSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(zdist_lpdf(x, mu, sigma, a, b));
    return rcpp_result_gen;
END_RCPP
}
// tdist_lpdf
NumericVector tdist_lpdf(NumericVector x, double mu, double df, double a, double b);
RcppExport SEXP _zcurve_tdist_lpdf(SEXP xSEXP, SEXP muSEXP, SEXP dfSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type df(dfSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(tdist_lpdf(x, mu, df, a, b));
    return rcpp_result_gen;
END_RCPP
}
// zdist_pdf
NumericVector zdist_pdf(NumericVector x, double mu, double sigma, double a, double b);
RcppExport SEXP _zcurve_zdist_pdf(SEXP xSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(zdist_pdf(x, mu, sigma, a, b));
    return rcpp_result_gen;
END_RCPP
}
// zdist_cens_lpdf
double zdist_cens_lpdf(double lb, double ub, double mu, double sigma, double a, double b);
RcppExport SEXP _zcurve_zdist_cens_lpdf(SEXP lbSEXP, SEXP ubSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< double >::type ub(ubSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(zdist_cens_lpdf(lb, ub, mu, sigma, a, b));
    return rcpp_result_gen;
END_RCPP
}
// tdist_pdf
NumericVector tdist_pdf(NumericVector x, double mu, double df, double a, double b);
RcppExport SEXP _zcurve_tdist_pdf(SEXP xSEXP, SEXP muSEXP, SEXP dfSEXP, SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type df(dfSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(tdist_pdf(x, mu, df, a, b));
    return rcpp_result_gen;
END_RCPP
}
// dirichlet_rng
NumericVector dirichlet_rng(NumericVector alpha);
RcppExport SEXP _zcurve_dirichlet_rng(SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(dirichlet_rng(alpha));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EM_fit_RCpp
List zcurve_EM_fit_RCpp(NumericVector x, int type, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EM_fit_RCpp(SEXP xSEXP, SEXP typeSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EM_fit_RCpp(x, type, mu, sigma, theta, a, b, sig_level, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EM_fit_fast_RCpp
List zcurve_EM_fit_fast_RCpp(NumericVector x, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EM_fit_fast_RCpp(SEXP xSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EM_fit_fast_RCpp(x, mu, sigma, theta, a, b, sig_level, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EMc_fit_fast_RCpp
List zcurve_EMc_fit_fast_RCpp(NumericVector x, NumericVector lb, NumericVector ub, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EMc_fit_fast_RCpp(SEXP xSEXP, SEXP lbSEXP, SEXP ubSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ub(ubSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EMc_fit_fast_RCpp(x, lb, ub, mu, sigma, theta, a, b, sig_level, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EMc_fit_fast_w_RCpp
List zcurve_EMc_fit_fast_w_RCpp(NumericVector x, NumericVector x_w, NumericVector lb, NumericVector ub, NumericVector b_w, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EMc_fit_fast_w_RCpp(SEXP xSEXP, SEXP x_wSEXP, SEXP lbSEXP, SEXP ubSEXP, SEXP b_wSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type x_w(x_wSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ub(ubSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type b_w(b_wSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EMc_fit_fast_w_RCpp(x, x_w, lb, ub, b_w, mu, sigma, theta, a, b, sig_level, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EM_start_RCpp
List zcurve_EM_start_RCpp(NumericVector x, int type, int K, NumericVector mu, NumericVector sigma, NumericVector mu_alpha, double mu_max, NumericVector theta_alpha, double a, double b, double sig_level, int fit_reps, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EM_start_RCpp(SEXP xSEXP, SEXP typeSEXP, SEXP KSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP mu_alphaSEXP, SEXP mu_maxSEXP, SEXP theta_alphaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP fit_repsSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu_alpha(mu_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type mu_max(mu_maxSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta_alpha(theta_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type fit_reps(fit_repsSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EM_start_RCpp(x, type, K, mu, sigma, mu_alpha, mu_max, theta_alpha, a, b, sig_level, fit_reps, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EM_boot_RCpp
List zcurve_EM_boot_RCpp(NumericVector x, int type, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int bootstrap, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EM_boot_RCpp(SEXP xSEXP, SEXP typeSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP bootstrapSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type type(typeSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type bootstrap(bootstrapSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EM_boot_RCpp(x, type, mu, sigma, theta, a, b, sig_level, bootstrap, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EM_start_fast_RCpp
List zcurve_EM_start_fast_RCpp(NumericVector x, int K, NumericVector mu, NumericVector sigma, NumericVector mu_alpha, double mu_max, NumericVector theta_alpha, double a, double b, double sig_level, int fit_reps, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EM_start_fast_RCpp(SEXP xSEXP, SEXP KSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP mu_alphaSEXP, SEXP mu_maxSEXP, SEXP theta_alphaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP fit_repsSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu_alpha(mu_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type mu_max(mu_maxSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta_alpha(theta_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type fit_reps(fit_repsSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EM_start_fast_RCpp(x, K, mu, sigma, mu_alpha, mu_max, theta_alpha, a, b, sig_level, fit_reps, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EM_boot_fast_RCpp
List zcurve_EM_boot_fast_RCpp(NumericVector x, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int bootstrap, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EM_boot_fast_RCpp(SEXP xSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP bootstrapSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type bootstrap(bootstrapSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EM_boot_fast_RCpp(x, mu, sigma, theta, a, b, sig_level, bootstrap, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EMc_start_fast_RCpp
List zcurve_EMc_start_fast_RCpp(NumericVector x, NumericVector lb, NumericVector ub, int K, NumericVector mu, NumericVector sigma, NumericVector mu_alpha, double mu_max, NumericVector theta_alpha, double a, double b, double sig_level, int fit_reps, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EMc_start_fast_RCpp(SEXP xSEXP, SEXP lbSEXP, SEXP ubSEXP, SEXP KSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP mu_alphaSEXP, SEXP mu_maxSEXP, SEXP theta_alphaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP fit_repsSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ub(ubSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu_alpha(mu_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type mu_max(mu_maxSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta_alpha(theta_alphaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type fit_reps(fit_repsSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EMc_start_fast_RCpp(x, lb, ub, K, mu, sigma, mu_alpha, mu_max, theta_alpha, a, b, sig_level, fit_reps, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EMc_boot_fast_RCpp
List zcurve_EMc_boot_fast_RCpp(NumericVector x, NumericVector lb, NumericVector ub, IntegerVector indx, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int bootstrap, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EMc_boot_fast_RCpp(SEXP xSEXP, SEXP lbSEXP, SEXP ubSEXP, SEXP indxSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP bootstrapSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ub(ubSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type indx(indxSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type bootstrap(bootstrapSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EMc_boot_fast_RCpp(x, lb, ub, indx, mu, sigma, theta, a, b, sig_level, bootstrap, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EMc_boot_fast_w_RCpp
List zcurve_EMc_boot_fast_w_RCpp(NumericVector x, NumericVector x_w, NumericVector lb, NumericVector ub, NumericVector b_w, IntegerVector indx, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int bootstrap, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EMc_boot_fast_w_RCpp(SEXP xSEXP, SEXP x_wSEXP, SEXP lbSEXP, SEXP ubSEXP, SEXP b_wSEXP, SEXP indxSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP bootstrapSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type x_w(x_wSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type lb(lbSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type ub(ubSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type b_w(b_wSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type indx(indxSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type bootstrap(bootstrapSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EMc_boot_fast_w_RCpp(x, x_w, lb, ub, b_w, indx, mu, sigma, theta, a, b, sig_level, bootstrap, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}
// zcurve_EM_PL_RCpp
List zcurve_EM_PL_RCpp(NumericVector x, NumericVector mu, NumericVector sigma, NumericVector theta, double a, double b, double sig_level, int max_iter, double criterion);
RcppExport SEXP _zcurve_zcurve_EM_PL_RCpp(SEXP xSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sig_levelSEXP, SEXP max_iterSEXP, SEXP criterionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type b(bSEXP);
    Rcpp::traits::input_parameter< double >::type sig_level(sig_levelSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type criterion(criterionSEXP);
    rcpp_result_gen = Rcpp::wrap(zcurve_EM_PL_RCpp(x, mu, sigma, theta, a, b, sig_level, max_iter, criterion));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_zcurve_zdist_lpdf", (DL_FUNC) &_zcurve_zdist_lpdf, 5},
    {"_zcurve_tdist_lpdf", (DL_FUNC) &_zcurve_tdist_lpdf, 5},
    {"_zcurve_zdist_pdf", (DL_FUNC) &_zcurve_zdist_pdf, 5},
    {"_zcurve_zdist_cens_lpdf", (DL_FUNC) &_zcurve_zdist_cens_lpdf, 6},
    {"_zcurve_tdist_pdf", (DL_FUNC) &_zcurve_tdist_pdf, 5},
    {"_zcurve_dirichlet_rng", (DL_FUNC) &_zcurve_dirichlet_rng, 1},
    {"_zcurve_zcurve_EM_fit_RCpp", (DL_FUNC) &_zcurve_zcurve_EM_fit_RCpp, 10},
    {"_zcurve_zcurve_EM_fit_fast_RCpp", (DL_FUNC) &_zcurve_zcurve_EM_fit_fast_RCpp, 9},
    {"_zcurve_zcurve_EMc_fit_fast_RCpp", (DL_FUNC) &_zcurve_zcurve_EMc_fit_fast_RCpp, 11},
    {"_zcurve_zcurve_EMc_fit_fast_w_RCpp", (DL_FUNC) &_zcurve_zcurve_EMc_fit_fast_w_RCpp, 13},
    {"_zcurve_zcurve_EM_start_RCpp", (DL_FUNC) &_zcurve_zcurve_EM_start_RCpp, 14},
    {"_zcurve_zcurve_EM_boot_RCpp", (DL_FUNC) &_zcurve_zcurve_EM_boot_RCpp, 11},
    {"_zcurve_zcurve_EM_start_fast_RCpp", (DL_FUNC) &_zcurve_zcurve_EM_start_fast_RCpp, 13},
    {"_zcurve_zcurve_EM_boot_fast_RCpp", (DL_FUNC) &_zcurve_zcurve_EM_boot_fast_RCpp, 10},
    {"_zcurve_zcurve_EMc_start_fast_RCpp", (DL_FUNC) &_zcurve_zcurve_EMc_start_fast_RCpp, 15},
    {"_zcurve_zcurve_EMc_boot_fast_RCpp", (DL_FUNC) &_zcurve_zcurve_EMc_boot_fast_RCpp, 13},
    {"_zcurve_zcurve_EMc_boot_fast_w_RCpp", (DL_FUNC) &_zcurve_zcurve_EMc_boot_fast_w_RCpp, 15},
    {"_zcurve_zcurve_EM_PL_RCpp", (DL_FUNC) &_zcurve_zcurve_EM_PL_RCpp, 9},
    {NULL, NULL, 0}
};

RcppExport void R_init_zcurve(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
