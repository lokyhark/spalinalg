use std::ops::Neg;

use crate::{scalar::Scalar, CsrMatrix};

impl<T: Scalar> Neg for &CsrMatrix<T> {
    type Output = CsrMatrix<T>;

    fn neg(self) -> Self::Output {
        let values: Vec<_> = self.values.iter().map(|&x| -x).collect();
        CsrMatrix {
            nrows: self.nrows(),
            ncols: self.ncols(),
            rowptr: self.rowptr.to_vec(),
            colind: self.colind.to_vec(),
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neg() {
        let mat = CsrMatrix::new(2, 1, vec![0, 1, 2], vec![0, 0], vec![1.0, 2.0]);
        let neg = -&mat;
        assert_eq!(neg.nrows, 2);
        assert_eq!(neg.ncols, 1);
        assert_eq!(neg.rowptr, [0, 1, 2]);
        assert_eq!(neg.colind, [0, 0]);
        assert_eq!(neg.values, [-1.0, -2.0]);
        assert_eq!(neg.rowptr.capacity(), neg.nrows() + 1);
        assert_eq!(neg.colind.capacity(), neg.nnz());
        assert_eq!(neg.values.capacity(), neg.nnz());
    }
}
