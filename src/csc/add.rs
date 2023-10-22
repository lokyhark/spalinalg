use std::ops::Add;

use crate::{scalar::Scalar, CscMatrix};

impl<T: Scalar> Add for &CscMatrix<T> {
    type Output = CscMatrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.nrows(), rhs.nrows());
        assert_eq!(self.ncols(), rhs.ncols());

        // Transpose inputs
        let (lhs, rhs) = (self.transpose(), rhs.transpose());

        // Allocate output
        let mut colptr = Vec::with_capacity(self.ncols() + 1);
        let cap = lhs.nnz() + rhs.nnz();
        let mut rowind = Vec::with_capacity(cap);
        let mut values = Vec::with_capacity(cap);

        // Allocate workspace
        let mut set = vec![0; lhs.nrows()];
        let mut vec = vec![T::zero(); lhs.nrows()];

        // Addition
        let mut nz = 0;
        for col in 0..lhs.ncols() {
            colptr.push(nz);
            for ptr in lhs.colptr[col]..lhs.colptr[col + 1] {
                let row = lhs.rowind[ptr];
                if set[row] < col + 1 {
                    set[row] = col + 1;
                    rowind.push(row);
                    vec[row] = lhs.values[ptr];
                    nz += 1;
                } else {
                    vec[row] += lhs.values[ptr];
                }
            }
            for ptr in rhs.colptr[col]..rhs.colptr[col + 1] {
                let row = rhs.rowind[ptr];
                if set[row] < col + 1 {
                    set[row] = col + 1;
                    rowind.push(row);
                    vec[row] = rhs.values[ptr];
                    nz += 1;
                } else {
                    vec[row] += rhs.values[ptr];
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
            nrows: self.nrows(),
            ncols: self.ncols(),
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
    fn add() {
        let lhs = CscMatrix::new(
            4,
            4,
            vec![0, 2, 4, 6, 7],
            vec![0, 1, 2, 3, 1, 3, 3],
            vec![1.0, 2.0, 4.0, 5.0, 3.0, 6.0, 7.0],
        );
        let rhs = CscMatrix::new(
            4,
            4,
            vec![0, 1, 2, 4, 5],
            vec![0, 3, 0, 1, 2],
            vec![2.0, 6.0, 4.0, 8.0, 10.0],
        );
        let mat = &lhs + &rhs;
        assert_eq!(mat.nrows, 4);
        assert_eq!(mat.ncols, 4);
        assert_eq!(mat.colptr, [0, 2, 4, 7, 9]);
        assert_eq!(mat.rowind, [0, 1, 2, 3, 0, 1, 3, 2, 3]);
        assert_eq!(mat.values, [3.0, 2.0, 4.0, 11.0, 4.0, 11.0, 6.0, 10.0, 7.0]);
        assert_eq!(mat.colptr.capacity(), mat.ncols() + 1);
        assert_eq!(mat.rowind.capacity(), mat.nnz());
        assert_eq!(mat.values.capacity(), mat.nnz());
    }
}
