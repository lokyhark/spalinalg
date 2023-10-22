use std::ops::Mul;

use crate::{scalar::Scalar, CscMatrix};

impl<T: Scalar> Mul for &CscMatrix<T> {
    type Output = CscMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.ncols(), rhs.nrows());

        // Transpose inputs
        let (lhs, rhs) = (rhs.transpose(), self.transpose());

        // Allocate output
        let mut colptr = Vec::with_capacity(rhs.ncols() + 1);
        let cap = lhs.nnz() + rhs.nnz();
        let mut rowind = Vec::with_capacity(cap);
        let mut values = Vec::with_capacity(cap);

        // Allocate workspace
        let mut set = vec![0; rhs.ncols()];
        let mut vec = vec![T::zero(); rhs.ncols()];

        // Multiply
        let mut nz = 0;
        for col in 0..rhs.ncols() {
            colptr.push(nz);
            for rhsptr in rhs.colptr[col]..rhs.colptr[col + 1] {
                let rhsrow = rhs.rowind[rhsptr];
                for lhsptr in lhs.colptr[rhsrow]..lhs.colptr[rhsrow + 1] {
                    let lhsrow = lhs.rowind[lhsptr];
                    if set[lhsrow] < col + 1 {
                        set[lhsrow] = col + 1;
                        rowind.push(lhsrow);
                        vec[lhsrow] = rhs.values[rhsptr] * lhs.values[lhsptr];
                        nz += 1;
                    } else {
                        vec[lhsrow] += rhs.values[rhsptr] * lhs.values[lhsptr];
                    }
                }
            }
            for ptr in colptr[col]..nz {
                let value = vec[rowind[ptr]];
                values.push(value)
            }
        }
        colptr.push(nz);

        // Construct matrix
        let output = CscMatrix {
            nrows: lhs.nrows(),
            ncols: rhs.ncols(),
            colptr,
            rowind,
            values,
        };

        // Transpose output
        output.transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mul() {
        let lhs = CscMatrix::new(
            5,
            3,
            vec![0, 3, 4, 6],
            vec![0, 1, 4, 3, 1, 2],
            vec![1.0, -5.0, 4.0, 3.0, 7.0, 2.0],
        );
        let rhs = CscMatrix::new(
            3,
            4,
            vec![0, 3, 4, 5, 6],
            vec![0, 1, 2, 2, 0, 1],
            vec![1.0, -5.0, 7.0, 3.0, -2.0, 4.0],
        );
        let mat = &lhs * &rhs;
        assert_eq!(mat.nrows, 5);
        assert_eq!(mat.ncols, 4);
        assert_eq!(mat.colptr, [0, 5, 7, 10, 11]);
        assert_eq!(mat.rowind, [0, 1, 2, 3, 4, 1, 2, 0, 1, 4, 3]);
        assert_eq!(
            mat.values,
            [1.0, 44.0, 14.0, -15.0, 4.0, 21.0, 6.0, -2.0, 10.0, -8.0, 12.0]
        );
        assert_eq!(mat.colptr.capacity(), mat.ncols() + 1);
        assert_eq!(mat.rowind.capacity(), mat.nnz());
        assert_eq!(mat.values.capacity(), mat.nnz());
    }
}
