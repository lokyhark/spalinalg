use crate::{scalar::Scalar, CooMatrix, CscMatrix};

impl<T: Scalar> From<&CooMatrix<T>> for CscMatrix<T> {
    fn from(coo: &CooMatrix<T>) -> Self {
        let nrows = coo.nrows();
        let ncols = coo.ncols();
        let len = coo.length();

        // Count number of entries in each row
        let mut vec = vec![0; nrows];
        for (row, _, _) in coo.iter() {
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
        let mut colind = vec![0; len];
        let mut rowval = vec![T::zero(); len];
        for (row, col, val) in coo.iter() {
            let ptr = &mut vec[row];
            colind[*ptr] = col;
            rowval[*ptr] = *val;
            *ptr += 1
        }

        // Sum up duplicates
        let mut vec = vec![None; ncols];
        let mut nz = 0;
        for row in 0..nrows {
            let start = nz;
            for ptr in rowptr[row]..rowptr[row + 1] {
                let col = colind[ptr];
                match vec[col] {
                    Some(prev) if prev >= start => {
                        let val = rowval[ptr];
                        rowval[prev] += val;
                    }
                    _ => {
                        vec[col] = Some(nz);
                        colind[nz] = col;
                        rowval[nz] = rowval[ptr];
                        nz += 1;
                    }
                }
            }
            rowptr[row] = start;
        }
        rowptr[nrows] = nz;

        // Drop numerically zero entries
        let mut nz = 0;
        for row in 0..nrows {
            let start = std::mem::replace(&mut rowptr[row], nz);
            for ptr in start..rowptr[row + 1] {
                if rowval[ptr] != T::zero() {
                    let ind = colind[ptr];
                    colind[nz] = ind;
                    let val = rowval[ptr];
                    rowval[nz] = val;
                    nz += 1;
                }
            }
        }
        rowptr[nrows] = nz;

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

impl<T: Scalar> From<CooMatrix<T>> for CscMatrix<T> {
    fn from(coo: CooMatrix<T>) -> Self {
        Self::from(&coo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_coo() {
        // | 1.0 + 2.0  3.0  4.0 |
        // |    0.0     0.0  5.0 |
        let mut coo = CooMatrix::new(2, 3);
        coo.push(1, 2, 5.0); // |
        coo.push(0, 2, 4.0); // |> unsorted rows
        coo.push(0, 1, 3.0); // |
        coo.push(0, 0, 1.0); // |> unsorted cols
        coo.push(0, 0, 2.0); // duplicate
        coo.push(1, 0, 0.0); // zero entry
        coo.push(1, 1, 1.00); // |
        coo.push(1, 1, -1.0); // |> numerical cancel
        let csc = CscMatrix::from(coo);
        assert_eq!(csc.colptr, [0, 1, 2, 4]);
        assert_eq!(csc.rowind, [0, 0, 0, 1]);
        assert_eq!(csc.values, [3.0, 3.0, 4.0, 5.0]);
    }
}
