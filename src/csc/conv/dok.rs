use crate::{scalar::Scalar, CscMatrix, DokMatrix};

impl<T: Scalar> From<&DokMatrix<T>> for CscMatrix<T> {
    fn from(dok: &DokMatrix<T>) -> Self {
        let nrows = dok.nrows();
        let ncols = dok.ncols();
        let nz = dok.length();

        // Count number of entries in each row
        let mut vec = vec![0; nrows];
        for (row, _, _) in dok.iter() {
            vec[row] += 1;
        }

        // Construct row pointers
        let mut rowptr = Vec::with_capacity(nrows + 1);
        let mut sum = 0;
        rowptr.push(0);
        for value in &mut vec {
            sum += *value;
            rowptr.push(sum);
        }

        // Construct compressed row form (rowptr, colind, values)
        let mut vec = rowptr[..nrows].to_vec();
        let mut colind = vec![0; nz];
        let mut rowval = vec![T::zero(); nz];
        for (row, col, val) in dok.iter() {
            let ptr = &mut vec[row];
            colind[*ptr] = col;
            rowval[*ptr] = *val;
            *ptr += 1
        }

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

impl<T: Scalar> From<DokMatrix<T>> for CscMatrix<T> {
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
        let csc = CscMatrix::from(dok);
        assert_eq!(csc.colptr, [0, 1, 2, 4]);
        assert_eq!(csc.rowind, [0, 0, 0, 1]);
        assert_eq!(csc.values, [3.0, 3.0, 4.0, 5.0]);
    }
}
