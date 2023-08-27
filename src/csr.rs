//! Compressed sparse row format module.

use crate::scalar::Scalar;

/// Compressed sparse row (CSR) format matrix.
#[derive(Debug)]
pub struct CsrMatrix<T: Scalar> {
    nrows: usize,
    ncols: usize,
    rowptr: Vec<usize>,
    colind: Vec<usize>,
    values: Vec<T>,
}

/// Immutable compressed sparse column matrix entries iterator created by [`CsrMatrix::iter`] method.
#[derive(Clone, Debug)]
pub struct Iter<'iter, T> {
    iter: std::vec::IntoIter<(usize, usize, &'iter T)>,
}

/// Mutable compressed sparse column matrix entries iterator created by [`CsrMatrix::iter_mut`] method.
#[derive(Debug)]
pub struct IterMut<'iter, T> {
    iter: std::vec::IntoIter<(usize, usize, &'iter mut T)>,
}

/// Move compressed sparse column matrix entries iterator created by [`CsrMatrix::into_iter`] method.
#[derive(Debug)]
pub struct IntoIter<T> {
    iter: std::vec::IntoIter<(usize, usize, T)>,
}

impl<T: Scalar> CsrMatrix<T> {
    pub fn new(
        nrows: usize,
        ncols: usize,
        rowptr: Vec<usize>,
        colind: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        assert!(nrows > 0);
        assert!(ncols > 0);
        assert!(rowptr.len() == nrows + 1);
        assert!(colind.len() == values.len());
        assert!(rowptr[0] == 0);
        assert!(rowptr.windows(2).all(|ptr| ptr[0] <= ptr[1]));
        assert!(colind.iter().all(|col| (0..ncols).contains(col)));
        Self {
            nrows,
            ncols,
            rowptr,
            colind,
            values,
        }
    }

    /// Returns number of rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.nrows(), 2);
    /// ```
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns number of rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.ncols(), 1);
    /// ```
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns row pointers.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.rowptr(), &[0, 1, 1]);
    /// ```
    pub fn rowptr(&self) -> &[usize] {
        &self.rowptr
    }

    /// Returns column indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.colind(), &[0]);
    /// ```
    pub fn colind(&self) -> &[usize] {
        &self.colind
    }

    /// Returns values slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.values(), &[1.0]);
    /// ```
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns mutable values slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let mut matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.values_mut(), &mut [1.0]);
    /// ```
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Returns number of non zeros in the matrix.
    ///
    /// # Examples
    /// /// Returns values slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.nnz(), 1);
    /// ```
    pub fn nnz(&self) -> usize {
        *self.rowptr.last().unwrap()
    }

    /// Returns an iterator over matrix entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// let mut iter = matrix.iter();
    /// assert_eq!(iter.next(), Some((0, 0, &1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter(&self) -> Iter<T> {
        let mut vec = Vec::with_capacity(self.nnz());
        let mut values = self.values.iter();
        for row in 0..self.nrows {
            for ptr in self.rowptr[row]..self.rowptr[row + 1] {
                let col = self.colind[ptr];
                let val = values.next().unwrap();
                vec.push((row, col, val));
            }
        }
        Iter {
            iter: vec.into_iter(),
        }
    }

    /// Returns a mutable iterator over matrix entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let mut matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// let mut iter = matrix.iter_mut();
    /// assert_eq!(iter.next(), Some((0, 0, &mut 1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<T> {
        let mut vec = Vec::with_capacity(self.nnz());
        let mut values = self.values.iter_mut();
        for row in 0..self.ncols {
            for ptr in self.rowptr[row]..self.rowptr[row + 1] {
                let col = self.colind[ptr];
                let val = values.next().unwrap();
                vec.push((row, col, val));
            }
        }
        IterMut {
            iter: vec.into_iter(),
        }
    }
}

impl<T: Scalar> IntoIterator for CsrMatrix<T> {
    type Item = (usize, usize, T);

    type IntoIter = IntoIter<T>;

    /// Turns compressed sparse column matrix into iterator over entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let matrix = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// let mut iter = matrix.into_iter();
    /// assert_eq!(iter.next(), Some((0, 0, 1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        let mut vec = Vec::with_capacity(self.nnz());
        let mut values = self.values.into_iter();
        for row in 0..self.nrows {
            for ptr in self.rowptr[row]..self.rowptr[row + 1] {
                let col = self.colind[ptr];
                let val = values.next().unwrap();
                vec.push((row, col, val));
            }
        }
        IntoIter {
            iter: vec.into_iter(),
        }
    }
}

impl<'iter, T> Iterator for Iter<'iter, T> {
    type Item = (usize, usize, &'iter T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'iter, T> Iterator for IterMut<'iter, T> {
    type Item = (usize, usize, &'iter mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<T: Scalar> Iterator for IntoIter<T> {
    type Item = (usize, usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn new_invalid_nrows() {
        CsrMatrix::<f64>::new(0, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_ncols() {
        CsrMatrix::<f64>::new(2, 0, vec![0, 1, 1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_colptr_first_not_zero() {
        CsrMatrix::<f64>::new(2, 1, vec![1, 1, 1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_colptr_invalid_length() {
        CsrMatrix::<f64>::new(2, 1, vec![0, 1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_rowind() {
        CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![1], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_rowind_values() {
        CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0, 2.0]);
    }
}
