use crate::{scalar::Scalar, CscMatrix, CsrMatrix};

impl<T: Scalar> From<&CsrMatrix<T>> for CscMatrix<T> {
    fn from(csr: &CsrMatrix<T>) -> Self {
        let nrows = csr.nrows();
        let ncols = csr.ncols();
        let rowptr = csr.rowptr();
        let colind = csr.colind();
        let rowval = csr.values();
        let nz = csr.nnz();

        // Count number of entries in each column
        let mut vec = vec![0; ncols];
        for row in 0..nrows {
            for ptr in rowptr[row]..rowptr[row + 1] {
                let col = colind[ptr];
                vec[col] += 1;
            }
        }

        // Construct column pointers
        let mut colptr = Vec::with_capacity(ncols + 1);
        let mut sum = 0;
        colptr.push(0);
        for value in vec {
            sum += value;
            colptr.push(sum);
        }

        // Construct column form
        let mut vec = colptr[..ncols].to_vec();
        let mut rowind = vec![0; nz];
        let mut colval = vec![T::zero(); nz];
        for row in 0..nrows {
            for ptr in rowptr[row]..rowptr[row + 1] {
                let col = colind[ptr];
                let idx = &mut vec[col];
                rowind[*idx] = row;
                colval[*idx] = rowval[ptr];
                *idx += 1;
            }
        }

        // Construct CscMatrix
        CscMatrix {
            nrows,
            ncols,
            colptr,
            rowind,
            values: colval,
        }
    }
}

impl<T: Scalar> From<CsrMatrix<T>> for CscMatrix<T> {
    fn from(csr: CsrMatrix<T>) -> Self {
        Self::from(&csr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_csr() {
        let csr = CsrMatrix::new(
            2,
            3,
            vec![0, 3, 4],
            vec![0, 1, 2, 2],
            vec![3.0, 3.0, 4.0, 5.0],
        );
        let csc = CscMatrix::from(csr);
        assert_eq!(csc.colptr, [0, 1, 2, 4]);
        assert_eq!(csc.rowind, [0, 0, 0, 1]);
        assert_eq!(csc.values, [3.0, 3.0, 4.0, 5.0]);
    }
}
