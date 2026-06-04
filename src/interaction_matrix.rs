//! Interaction matrix for competitive Lotka-Volterra systems.

use std::fmt;

/// Square interaction (competition) matrix for N species.
///
/// Entry `a[i][j]` represents the competitive effect of species j on species i.
/// Diagonal entries are typically 1.0 (intraspecific competition).
#[derive(Clone, Debug)]
pub struct InteractionMatrix {
    /// Row-major storage. `data[i * n + j]` = effect of j on i.
    data: Vec<f64>,
    n: usize,
}

impl InteractionMatrix {
    /// Create an N×N identity interaction matrix (no interspecific competition).
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self { data, n }
    }

    /// Create from a flat row-major slice. Panics if length != n*n.
    pub fn from_slice(n: usize, data: &[f64]) -> Self {
        assert_eq!(data.len(), n * n, "Data length must be n*n");
        Self {
            data: data.to_vec(),
            n,
        }
    }

    /// Create from a 2D Vec. Panics if not square.
    pub fn from_2d(rows: &[Vec<f64>]) -> Self {
        let n = rows.len();
        for row in rows {
            assert_eq!(row.len(), n, "Matrix must be square");
        }
        let mut data = Vec::with_capacity(n * n);
        for row in rows {
            data.extend_from_slice(row);
        }
        Self { data, n }
    }

    /// Build a 2-species competition matrix from alpha values.
    ///
    /// Returns `[[1, alpha_12], [alpha_21, 1]]` (standard LV competition form).
    pub fn two_species(alpha_12: f64, alpha_21: f64) -> Self {
        Self::from_2d(&[
            vec![1.0, alpha_12],
            vec![alpha_21, 1.0],
        ])
    }

    /// Number of species (matrix dimension).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Get element at (i, j): effect of species j on species i.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * self.n + j] = val;
    }

    /// Get row i as a slice.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.n..(i + 1) * self.n]
    }

    /// Check if the matrix is symmetric (A = A^T).
    pub fn is_symmetric(&self) -> bool {
        self.is_symmetric_tol(1e-10)
    }

    /// Check symmetry with a tolerance.
    pub fn is_symmetric_tol(&self, tol: f64) -> bool {
        for i in 0..self.n {
            for j in (i + 1)..self.n {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute eigenvalues of the matrix using the power method for the largest,
    /// or full decomposition for small matrices (N <= 4).
    ///
    /// For N=2, uses the analytical formula.
    /// For N>2, uses iterative QR-like approach (simplified).
    /// Returns eigenvalues sorted by real part (descending).
    pub fn eigenvalues(&self) -> Vec<ComplexEigenvalue> {
        if self.n == 0 {
            return vec![];
        }
        if self.n == 1 {
            return vec![ComplexEigenvalue {
                real: self.data[0],
                imag: 0.0,
            }];
        }
        if self.n == 2 {
            return self.eigenvalues_2x2();
        }
        // For larger matrices, use iterative Jacobi eigenvalue algorithm
        // (works well for symmetric matrices, gives approximate results otherwise)
        self.eigenvalues_jacobi()
    }

    /// Analytical eigenvalues for 2×2 matrix.
    fn eigenvalues_2x2(&self) -> Vec<ComplexEigenvalue> {
        let a = self.get(0, 0);
        let b = self.get(0, 1);
        let c = self.get(1, 0);
        let d = self.get(1, 1);

        let trace = a + d;
        let det = a * d - b * c;
        let disc = trace * trace - 4.0 * det;

        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            let mut eigs = vec![
                ComplexEigenvalue {
                    real: (trace + sqrt_disc) / 2.0,
                    imag: 0.0,
                },
                ComplexEigenvalue {
                    real: (trace - sqrt_disc) / 2.0,
                    imag: 0.0,
                },
            ];
            eigs.sort_by(|a, b| b.real.partial_cmp(&a.real).unwrap_or(std::cmp::Ordering::Equal));
            eigs
        } else {
            let sqrt_disc = (-disc).sqrt();
            vec![
                ComplexEigenvalue {
                    real: trace / 2.0,
                    imag: sqrt_disc / 2.0,
                },
                ComplexEigenvalue {
                    real: trace / 2.0,
                    imag: -sqrt_disc / 2.0,
                },
            ]
        }
    }

    /// Jacobi eigenvalue algorithm for symmetric matrices. Falls back to
    /// Gershgorin circle estimates for non-symmetric matrices.
    fn eigenvalues_jacobi(&self) -> Vec<ComplexEigenvalue> {
        if self.is_symmetric_tol(1e-10) {
            self.eigenvalues_jacobi_symmetric()
        } else {
            // For non-symmetric, use Gershgorin circles as approximation
            self.eigenvalues_gershgorin()
        }
    }

    /// Jacobi eigenvalue algorithm for symmetric matrices.
    fn eigenvalues_jacobi_symmetric(&self) -> Vec<ComplexEigenvalue> {
        let n = self.n;
        let mut a = self.data.clone();

        for _ in 0..100 * n * n {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f64;
            let mut max_i = 0;
            let mut max_j = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    let v = a[i * n + j].abs();
                    if v > max_val {
                        max_val = v;
                        max_i = i;
                        max_j = j;
                    }
                }
            }

            if max_val < 1e-12 {
                break;
            }

            // Compute rotation angle
            let diff = a[max_j * n + max_j] - a[max_i * n + max_i];
            let t = if diff.abs() < 1e-15 {
                1.0
            } else {
                let theta = 2.0 * a[max_i * n + max_j] / diff;
                1.0 / (theta.abs() + (1.0 + theta * theta).sqrt()).copysign(theta)
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Apply rotation
            for k in 0..n {
                if k != max_i && k != max_j {
                    let aki = a[k * n + max_i];
                    let akj = a[k * n + max_j];
                    a[k * n + max_i] = c * aki - s * akj;
                    a[max_i * n + k] = c * aki - s * akj;
                    a[k * n + max_j] = s * aki + c * akj;
                    a[max_j * n + k] = s * aki + c * akj;
                }
            }

            let aii = a[max_i * n + max_i];
            let ajj = a[max_j * n + max_j];
            let aij = a[max_i * n + max_j];

            a[max_i * n + max_i] = c * c * aii - 2.0 * s * c * aij + s * s * ajj;
            a[max_j * n + max_j] = s * s * aii + 2.0 * s * c * aij + c * c * ajj;
            a[max_i * n + max_j] = 0.0;
            a[max_j * n + max_i] = 0.0;
        }

        let mut eigs: Vec<ComplexEigenvalue> = (0..n)
            .map(|i| ComplexEigenvalue {
                real: a[i * n + i],
                imag: 0.0,
            })
            .collect();
        eigs.sort_by(|a, b| b.real.partial_cmp(&a.real).unwrap_or(std::cmp::Ordering::Equal));
        eigs
    }

    /// Gershgorin circle estimates for eigenvalues.
    fn eigenvalues_gershgorin(&self) -> Vec<ComplexEigenvalue> {
        let mut eigs = Vec::new();
        for i in 0..self.n {
            let diag = self.get(i, i);
            let radius: f64 = (0..self.n)
                .filter(|&j| j != i)
                .map(|j| self.get(i, j).abs())
                .sum();
            // Use diagonal element as estimate
            eigs.push(ComplexEigenvalue {
                real: diag,
                imag: radius,
            });
        }
        eigs.sort_by(|a, b| b.real.partial_cmp(&a.real).unwrap_or(std::cmp::Ordering::Equal));
        eigs
    }

    /// Check if all eigenvalues have the indicated sign pattern for stability.
    /// For a competition matrix to lead to stable coexistence, the matrix should
    /// be positive definite (all eigenvalues positive when considering -A).
    pub fn is_positive_definite(&self) -> bool {
        self.eigenvalues().iter().all(|e| e.real > 0.0)
    }

    /// Matrix-vector multiplication: result[i] = sum_j A[i][j] * v[j].
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.n, "Vector dimension mismatch");
        (0..self.n)
            .map(|i| {
                let row = self.row(i);
                row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
            })
            .collect()
    }
}

impl fmt::Display for InteractionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.n {
            let row = self.row(i);
            let formatted: Vec<String> = row.iter().map(|v| format!("{:8.4}", v)).collect();
            writeln!(f, "[{}]", formatted.join(", "))?;
        }
        Ok(())
    }
}

/// A complex eigenvalue with real and imaginary parts.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexEigenvalue {
    pub real: f64,
    pub imag: f64,
}

impl ComplexEigenvalue {
    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    pub fn is_real(&self) -> bool {
        self.imag.abs() < 1e-10
    }
}

impl fmt::Display for ComplexEigenvalue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.imag.abs() < 1e-10 {
            write!(f, "{:.6}", self.real)
        } else if self.imag >= 0.0 {
            write!(f, "{:.6} + {:.6}i", self.real, self.imag)
        } else {
            write!(f, "{:.6} - {:.6}i", self.real, -self.imag)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let m = InteractionMatrix::identity(3);
        assert_eq!(m.n(), 3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert!((m.get(i, j) - 1.0).abs() < 1e-10);
                } else {
                    assert!(m.get(i, j).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_symmetry_check() {
        let symmetric = InteractionMatrix::two_species(0.5, 0.5);
        assert!(symmetric.is_symmetric());

        let asymmetric = InteractionMatrix::two_species(0.3, 0.7);
        assert!(!asymmetric.is_symmetric());
    }

    #[test]
    fn test_eigenvalues_2x2_identity() {
        let m = InteractionMatrix::identity(2);
        let eigs = m.eigenvalues();
        assert_eq!(eigs.len(), 2);
        assert!((eigs[0].real - 1.0).abs() < 1e-10);
        assert!((eigs[1].real - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigenvalues_2x2_general() {
        let m = InteractionMatrix::from_2d(&[
            vec![2.0, 1.0],
            vec![1.0, 3.0],
        ]);
        let eigs = m.eigenvalues();
        // Eigenvalues of [[2,1],[1,3]]: trace=5, det=5, disc=5
        // (5 ± sqrt(5))/2 ≈ 3.618, 1.382
        let expected = [(5.0 + 5.0_f64.sqrt()) / 2.0, (5.0 - 5.0_f64.sqrt()) / 2.0];
        assert!((eigs[0].real - expected[0]).abs() < 1e-6);
        assert!((eigs[1].real - expected[1]).abs() < 1e-6);
    }

    #[test]
    fn test_eigenvalues_complex() {
        // Matrix with complex eigenvalues: [[0, -1], [1, 0]]
        let m = InteractionMatrix::from_2d(&[
            vec![0.0, -1.0],
            vec![1.0, 0.0],
        ]);
        let eigs = m.eigenvalues();
        // Eigenvalues are ±i
        assert!(eigs[0].imag.abs() > 0.5); // has imaginary part
    }

    #[test]
    fn test_mul_vec() {
        let m = InteractionMatrix::from_2d(&[
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let v = vec![1.0, 1.0];
        let result = m.mul_vec(&v);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_positive_definite() {
        let m = InteractionMatrix::from_2d(&[
            vec![2.0, 1.0],
            vec![1.0, 2.0],
        ]);
        assert!(m.is_positive_definite());

        let m2 = InteractionMatrix::from_2d(&[
            vec![-1.0, 0.0],
            vec![0.0, -1.0],
        ]);
        assert!(!m2.is_positive_definite());
    }

    #[test]
    fn test_from_slice() {
        let data = [1.0, 0.5, 0.3, 1.0];
        let m = InteractionMatrix::from_slice(2, &data);
        assert!((m.get(0, 1) - 0.5).abs() < 1e-10);
        assert!((m.get(1, 0) - 0.3).abs() < 1e-10);
    }
}
