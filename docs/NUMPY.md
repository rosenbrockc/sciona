# NumPy Ingestion Audit: Valuable New Targets

This list identifies high-value algorithmic targets in NumPy that represent core computational logic (beyond basic arithmetic or element-wise ops) and are missing from the current `ageoa` catalog.

---

## 1. Linear Algebra (`numpy.linalg`)
*Core decompositions and advanced solvers.*

| Target | Description |
|---|---|
| `numpy.linalg.eig` | Compute eigenvalues and right eigenvectors of a square array. |
| `numpy.linalg.eigh` | Return the eigenvalues and eigenvectors of a complex Hermitian or real symmetric matrix. |
| `numpy.linalg.pinv` | Compute the Moore-Penrose pseudo-inverse of a matrix. |
| `numpy.linalg.lstsq` | Return the least-squares solution to a linear matrix equation. |
| `numpy.linalg.matrix_rank` | Return matrix rank of array using SVD method. |
| `numpy.linalg.qr` | Compute the qr factorization of a matrix. |
| `numpy.linalg.cond` | Compute the condition number of a matrix. |

## 2. Fast Fourier Transform (`numpy.fft`)
*Multidimensional and symmetry-optimized transforms.*

| Target | Description |
|---|---|
| `numpy.fft.fft2` | Compute the 2-dimensional discrete Fourier Transform. |
| `numpy.fft.ifft2` | Compute the 2-dimensional inverse discrete Fourier Transform. |
| `numpy.fft.fftn` | Compute the N-dimensional discrete Fourier Transform. |
| `numpy.fft.ifftn` | Compute the N-dimensional inverse discrete Fourier Transform. |
| `numpy.fft.hfft` | Compute the FFT of a signal that has Hermitian symmetry. |
| `numpy.fft.ihfft` | Compute the inverse FFT of a signal that has Hermitian symmetry. |

## 3. Mathematical Routines
*Complex coordination and specialized logic.*

| Target | Description |
|---|---|
| `numpy.einsum` | Evaluates the Einstein summation convention on the operands. |
| `numpy.einsum_path` | Evaluates the lowest-cost contraction order for an einsum expression. |
| `numpy.unique` | Find the unique elements of an array. |
| `numpy.interp` | One-dimensional linear interpolation for monotonically increasing sample points. |
| `numpy.gradient` | Return the gradient of an N-dimensional array. |
| `numpy.cross` | Return the cross product of two (arrays of) vectors. |
| `numpy.cov` | Estimate a covariance matrix, given data and weights. |
| `numpy.corrcoef` | Return Pearson product-moment correlation coefficients. |

## 4. Sorting, Searching, and Counting
*Optimized C-implemented search/sort logic.*

| Target | Description |
|---|---|
| `numpy.searchsorted` | Find indices where elements should be inserted to maintain order. |
| `numpy.lexsort` | Indirect stable sort on multiple keys. |
| `numpy.partition` | Return a partitioned copy of an array (partial sort). |
| `numpy.argpartition` | Perform an indirect partial sort. |

## 5. Random Sampling (`numpy.random`)
*Modern Generator-based distributions.*

| Target | Description |
|---|---|
| `numpy.random.Generator.multivariate_normal` | Draw random samples from a multivariate normal distribution. |
| `numpy.random.Generator.dirichlet` | Draw samples from the Dirichlet distribution. |
| `numpy.random.Generator.multinomial` | Draw samples from the multinomial distribution. |
| `numpy.random.Generator.permutation` | Randomly permute a sequence, or return a permuted range. |
| `numpy.random.Generator.choice` | Generates a random sample from a given 1-D array. |
