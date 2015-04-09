///
/// If the estd library is installed, allow
/// its use within ecuda.  This allows the
/// estd::matrix and estd::cube classes to
/// be used with ecuda::matrix and ecuda::cube.
///
#define HAVE_ESTD_LIBRARY 0

///
/// If the GNU Scientific Library (GSL) is
/// installed, allow its use within ecuda.
/// This allows the estd::matrix and estd::cube
/// classes to utilize gsl_matrix.
#define HAVE_GNU_SCIENTIFIC_LIBRARY 0
