use std::ops::Mul;

use crate::{scalar::Scalar, CsrMatrix};

impl<T: Scalar> Mul for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.ncols(), rhs.nrows());
        // Transpose inputs
        let (lhs, rhs) = (rhs.transpose(), self.transpose());

        // Allocate output
        let mut rowptr = Vec::with_capacity(lhs.nrows() + 1);
        let cap = lhs.nnz() + rhs.nnz();
        let mut colind = Vec::with_capacity(cap);
        let mut values = Vec::with_capacity(cap);

        // Allocate workspace
        let mut set = vec![0; rhs.ncols()];
        let mut vec = vec![T::zero(); rhs.ncols()];

        // Multiply
        let mut nz = 0;
        for row in 0..lhs.nrows() {
            rowptr.push(nz);
            for lhsptr in lhs.rowptr[row]..lhs.rowptr[row + 1] {
                let lhscol = lhs.colind[lhsptr];
                for rhsptr in rhs.rowptr[lhscol]..rhs.rowptr[lhscol + 1] {
                    let rhscol = rhs.colind[rhsptr];
                    if set[rhscol] < row + 1 {
                        set[rhscol] = row + 1;
                        colind.push(rhscol);
                        vec[rhscol] = rhs.values[rhsptr] * lhs.values[lhsptr];
                        nz += 1;
                    } else {
                        vec[rhscol] += rhs.values[rhsptr] * lhs.values[lhsptr];
                    }
                }
            }
            for ptr in rowptr[row]..nz {
                let value = vec[colind[ptr]];
                values.push(value)
            }
        }
        rowptr.push(nz);

        // Construct matrix
        let output = CsrMatrix {
            nrows: lhs.nrows(),
            ncols: rhs.ncols(),
            rowptr,
            colind,
            values,
        };

        // Transpose output
        output.transpose()
    }
}
