use std::ops::Neg;

use crate::{scalar::Scalar, CscMatrix};

impl<T: Scalar> Neg for &CscMatrix<T> {
    type Output = CscMatrix<T>;

    fn neg(self) -> Self::Output {
        let values: Vec<_> = self.values.iter().map(|&x| -x).collect();
        CscMatrix {
            nrows: self.nrows(),
            ncols: self.ncols(),
            colptr: self.colptr.to_vec(),
            rowind: self.rowind.to_vec(),
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg() {
        let mat = CscMatrix::new(1, 2, vec![0, 1, 2], vec![0, 0], vec![1.0, 2.0]);
        let neg = -&mat;
        assert_eq!(neg.nrows, 1);
        assert_eq!(neg.ncols, 2);
        assert_eq!(neg.colptr, [0, 1, 2]);
        assert_eq!(neg.rowind, [0, 0]);
        assert_eq!(neg.values, [-1.0, -2.0]);
        assert_eq!(neg.colptr.capacity(), neg.ncols() + 1);
        assert_eq!(neg.rowind.capacity(), neg.nnz());
        assert_eq!(neg.values.capacity(), neg.nnz());
    }
}
