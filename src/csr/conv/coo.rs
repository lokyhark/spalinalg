use crate::{scalar::Scalar, CooMatrix, CsrMatrix};

impl<T: Scalar> From<&CooMatrix<T>> for CsrMatrix<T> {
    fn from(coo: &CooMatrix<T>) -> Self {
        let nrows = coo.nrows();
        let ncols = coo.ncols();
        let len = coo.length();

        // Count number of entries in each row
        let mut vec = vec![0; ncols];
        for (_, col, _) in coo.iter() {
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
        let mut rowind = vec![0; len];
        let mut colval = vec![T::zero(); len];
        for (row, col, val) in coo.iter() {
            let ptr = &mut vec[col];
            rowind[*ptr] = row;
            colval[*ptr] = *val;
            *ptr += 1
        }

        // Sum up duplicates
        let mut vec = vec![None; nrows];
        let mut nz = 0;
        for col in 0..ncols {
            let start = nz;
            for ptr in colptr[col]..colptr[col + 1] {
                let row = rowind[ptr];
                match vec[row] {
                    Some(prev) if prev >= start => {
                        let val = colval[ptr];
                        colval[prev] += val;
                    }
                    _ => {
                        vec[row] = Some(nz);
                        rowind[nz] = row;
                        colval[nz] = colval[ptr];
                        nz += 1;
                    }
                }
            }
            colptr[col] = start;
        }
        colptr[ncols] = nz;

        // Drop numerically zero entries
        let mut nz = 0;
        for col in 0..ncols {
            let start = std::mem::replace(&mut colptr[col], nz);
            for ptr in start..colptr[col + 1] {
                if colval[ptr] != T::zero() {
                    let ind = rowind[ptr];
                    rowind[nz] = ind;
                    let val = colval[ptr];
                    colval[nz] = val;
                    nz += 1;
                }
            }
        }
        colptr[ncols] = nz;

        // Count number of entries in each row
        let mut vec = vec![0; nrows];
        for col in 0..ncols {
            for ptr in colptr[col]..colptr[col + 1] {
                let row = rowind[ptr];
                vec[row] += 1
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

impl<T: Scalar> From<CooMatrix<T>> for CsrMatrix<T> {
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
        let csr = CsrMatrix::from(coo);
        assert_eq!(csr.rowptr, [0, 3, 4]);
        assert_eq!(csr.colind, [0, 1, 2, 2]);
        assert_eq!(csr.values, [3.0, 3.0, 4.0, 5.0]);
    }
}
