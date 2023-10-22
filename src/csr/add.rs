use std::ops::Add;

use crate::{scalar::Scalar, CsrMatrix};

impl<T: Scalar> Add for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.nrows(), rhs.nrows());
        assert_eq!(self.ncols(), rhs.ncols());

        // Transpose inputs
        let (lhs, rhs) = (self.transpose(), rhs.transpose());
        dbg!(&lhs);
        dbg!(&rhs);

        // Allocate output
        let mut rowptr = Vec::with_capacity(self.nrows() + 1);
        let cap = lhs.nnz() + rhs.nnz();
        let mut colind = Vec::with_capacity(cap);
        let mut values = Vec::with_capacity(cap);

        // Allocate workspace
        let mut set = vec![0; lhs.ncols()];
        let mut vec = vec![T::zero(); lhs.ncols()];

        // Addition
        let mut nz = 0;
        for row in 0..lhs.nrows() {
            rowptr.push(nz);
            for ptr in lhs.rowptr[row]..lhs.rowptr[row + 1] {
                let col = lhs.colind[ptr];
                if set[col] < row + 1 {
                    set[col] = row + 1;
                    colind.push(col);
                    vec[col] = lhs.values[ptr];
                    nz += 1;
                } else {
                    vec[col] += lhs.values[ptr];
                }
            }
            for ptr in rhs.rowptr[row]..rhs.rowptr[row + 1] {
                let col = rhs.colind[ptr];
                if set[col] < row + 1 {
                    set[col] = row + 1;
                    colind.push(col);
                    vec[col] = rhs.values[ptr];
                    nz += 1;
                } else {
                    vec[col] += rhs.values[ptr];
                }
            }
            for ptr in rowptr[row]..nz {
                let value = vec[colind[ptr]];
                values.push(value)
            }
            dbg!(&rowptr);
            dbg!(&colind);
            dbg!(&values);
        }
        rowptr.push(nz);

        // Construct matrix
        let output = CsrMatrix {
            nrows: self.nrows(),
            ncols: self.ncols(),
            rowptr,
            colind,
            values,
        };

        // Transpose output
        dbg!(output.transpose())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let lhs = CsrMatrix::new(
            4,
            4,
            vec![0, 1, 3, 4, 7],
            vec![0, 0, 2, 1, 1, 2, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        );
        let rhs = CsrMatrix::new(
            4,
            4,
            vec![0, 2, 3, 4, 5],
            vec![0, 2, 2, 3, 1],
            vec![2.0, 4.0, 8.0, 10.0, 6.0],
        );
        let mat = &lhs + &rhs;
        assert_eq!(mat.nrows, 4);
        assert_eq!(mat.ncols, 4);
        assert_eq!(mat.rowptr, [0, 2, 4, 6, 9]);
        assert_eq!(mat.colind, [0, 2, 0, 2, 1, 3, 1, 2, 3]);
        assert_eq!(mat.values, [3.0, 4.0, 2.0, 11.0, 4.0, 10.0, 11.0, 6.0, 7.0]);
        assert_eq!(mat.rowptr.capacity(), mat.nrows() + 1);
        assert_eq!(mat.colind.capacity(), mat.nnz());
        assert_eq!(mat.values.capacity(), mat.nnz());
    }
}
