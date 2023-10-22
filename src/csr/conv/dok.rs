use crate::{scalar::Scalar, CsrMatrix, DokMatrix};

impl<T: Scalar> From<&DokMatrix<T>> for CsrMatrix<T> {
    fn from(dok: &DokMatrix<T>) -> Self {
        let nrows = dok.nrows();
        let ncols = dok.ncols();
        let nz = dok.length();

        // Count number of entries in each row
        let mut vec = vec![0; ncols];
        for (_, col, _) in dok.iter() {
            vec[col] += 1;
        }

        // Construct col pointers
        let mut colptr = Vec::with_capacity(ncols + 1);
        let mut sum = 0;
        colptr.push(0);
        for value in &mut vec {
            sum += *value;
            colptr.push(sum);
        }

        // Construct compressed row form (colptr, rowind, colval)
        let mut vec = colptr[..ncols].to_vec();
        let mut rowind = vec![0; nz];
        let mut colval = vec![T::zero(); nz];
        for (row, col, val) in dok.iter() {
            let ptr = &mut vec[col];
            rowind[*ptr] = row;
            colval[*ptr] = *val;
            *ptr += 1
        }

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

impl<T: Scalar> From<DokMatrix<T>> for CsrMatrix<T> {
    fn from(dok: DokMatrix<T>) -> Self {
        Self::from(&dok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_dok() {
        let mut dok = DokMatrix::new(2, 3);
        dok.insert(0, 0, 3.0);
        dok.insert(0, 1, 3.0);
        dok.insert(0, 2, 4.0);
        dok.insert(1, 2, 5.0);
        let csr = CsrMatrix::from(dok);
        assert_eq!(csr.rowptr, [0, 3, 4]);
        assert_eq!(csr.colind, [0, 1, 2, 2]);
        assert_eq!(csr.values, [3.0, 3.0, 4.0, 5.0]);
    }
}
