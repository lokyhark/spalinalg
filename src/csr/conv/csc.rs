use crate::{scalar::Scalar, CscMatrix, CsrMatrix};

impl<T: Scalar> From<&CscMatrix<T>> for CsrMatrix<T> {
    fn from(csc: &CscMatrix<T>) -> Self {
        let nrows = csc.nrows();
        let ncols = csc.ncols();
        let colptr = csc.colptr();
        let rowind = csc.rowind();
        let colval = csc.values();
        let nz = csc.nnz();

        // Count number of entries in each row
        let mut vec = vec![0; nrows];
        for col in 0..ncols {
            for ptr in colptr[col]..colptr[col + 1] {
                let row = rowind[ptr];
                vec[row] += 1;
            }
        }

        // Construct row pointers
        let mut rowptr = Vec::with_capacity(nrows + 1);
        let mut sum = 0;
        rowptr.push(0);
        for value in vec {
            sum += value;
            rowptr.push(sum);
        }

        // Construct row form
        let mut vec = rowptr[..nrows].to_vec();
        let mut colind = vec![0; nz];
        let mut rowval = vec![T::zero(); nz];
        for col in 0..ncols {
            for ptr in colptr[col]..colptr[col + 1] {
                let row = rowind[ptr];
                let idx = &mut vec[row];
                colind[*idx] = col;
                rowval[*idx] = colval[ptr];
                *idx += 1;
            }
        }

        // Construct CsrMatrix
        CsrMatrix {
            nrows,
            ncols,
            rowptr,
            colind,
            values: rowval,
        }
    }
}

impl<T: Scalar> From<CscMatrix<T>> for CsrMatrix<T> {
    fn from(csc: CscMatrix<T>) -> Self {
        Self::from(&csc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_csc() {
        let csc = CscMatrix::new(
            2,
            3,
            vec![0, 1, 2, 4],
            vec![0, 0, 0, 1],
            vec![3.0, 3.0, 4.0, 5.0],
        );
        let csr = CsrMatrix::from(csc);
        assert_eq!(csr.rowptr, [0, 3, 4]);
        assert_eq!(csr.colind, [0, 1, 2, 2]);
        assert_eq!(csr.values, [3.0, 3.0, 4.0, 5.0]);
    }
}
