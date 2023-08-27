//! Compressed sparse column format module.

/// Compressed sparse column (CSC) format matrix.
#[derive(Debug)]
pub struct CscMatrix<T> {
    nrows: usize,
    ncols: usize,
    colptr: Vec<usize>,
    rowind: Vec<usize>,
    values: Vec<T>,
}

/// Immutable compressed sparse column matrix entries iterator created by [`CscMatrix::iter`] method.
#[derive(Clone, Debug)]
pub struct Iter<'iter, T> {
    iter: std::vec::IntoIter<(usize, usize, &'iter T)>,
}

/// Mutable compressed sparse column matrix entries iterator created by [`CscMatrix::iter_mut`] method.
#[derive(Debug)]
pub struct IterMut<'iter, T> {
    iter: std::vec::IntoIter<(usize, usize, &'iter mut T)>,
}

/// Move compressed sparse column matrix entries iterator created by [`CscMatrix::into_iter`] method.
#[derive(Debug)]
pub struct IntoIter<T> {
    iter: std::vec::IntoIter<(usize, usize, T)>,
}

impl<T> CscMatrix<T> {
    pub fn new(
        nrows: usize,
        ncols: usize,
        colptr: Vec<usize>,
        rowind: Vec<usize>,
        values: Vec<T>,
    ) -> Self {
        assert!(nrows > 0);
        assert!(ncols > 0);
        assert!(colptr.len() == ncols + 1);
        assert!(rowind.len() == values.len());
        assert!(colptr[0] == 0);
        assert!(colptr.windows(2).all(|ptr| ptr[0] <= ptr[1]));
        assert!(rowind.iter().all(|row| (0..nrows).contains(row)));
        Self {
            nrows,
            ncols,
            colptr,
            rowind,
            values,
        }
    }

    /// Returns number of rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    ///
    /// let matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.nrows(), 1);
    /// ```
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Returns number of rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    ///
    /// let matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.ncols(), 2);
    /// ```
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns column pointers.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    ///
    /// let matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.colptr(), &[0, 1, 1]);
    /// ```
    pub fn colptr(&self) -> &[usize] {
        &self.colptr
    }

    /// Returns row indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    ///
    /// let matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.rowind(), &[0]);
    /// ```
    pub fn rowind(&self) -> &[usize] {
        &self.rowind
    }

    /// Returns values slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    ///
    /// let matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
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
    /// use spalinalg::CscMatrix;
    ///
    /// let mut matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
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
    /// use spalinalg::CscMatrix;
    ///
    /// let matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// assert_eq!(matrix.nnz(), 1);
    /// ```
    pub fn nnz(&self) -> usize {
        *self.colptr.last().unwrap()
    }

    /// Returns an iterator over matrix entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    ///
    /// let matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// let mut iter = matrix.iter();
    /// assert_eq!(iter.next(), Some((0, 0, &1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter(&self) -> Iter<T> {
        let mut vec = Vec::with_capacity(self.nnz());
        let mut values = self.values.iter();
        for col in 0..self.ncols {
            for ptr in self.colptr[col]..self.colptr[col + 1] {
                let row = self.rowind[ptr];
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
    /// use spalinalg::CscMatrix;
    ///
    /// let mut matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// let mut iter = matrix.iter_mut();
    /// assert_eq!(iter.next(), Some((0, 0, &mut 1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<T> {
        let mut vec = Vec::with_capacity(self.nnz());
        let mut values = self.values.iter_mut();
        for col in 0..self.ncols {
            for ptr in self.colptr[col]..self.colptr[col + 1] {
                let row = self.rowind[ptr];
                let val = values.next().unwrap();
                vec.push((row, col, val));
            }
        }
        IterMut {
            iter: vec.into_iter(),
        }
    }
}

impl<T> IntoIterator for CscMatrix<T> {
    type Item = (usize, usize, T);

    type IntoIter = IntoIter<T>;

    /// Turns compressed sparse column matrix into iterator over entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    ///
    /// let matrix = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// let mut iter = matrix.into_iter();
    /// assert_eq!(iter.next(), Some((0, 0, 1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        let mut vec = Vec::with_capacity(self.nnz());
        let mut values = self.values.into_iter();
        for col in 0..self.ncols {
            for ptr in self.colptr[col]..self.colptr[col + 1] {
                let row = self.rowind[ptr];
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

impl<T> Iterator for IntoIter<T> {
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
        CscMatrix::<f64>::new(0, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_ncols() {
        CscMatrix::<f64>::new(1, 0, vec![0, 1, 1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_colptr_first_not_zero() {
        CscMatrix::<f64>::new(1, 2, vec![1, 1, 1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_colptr_invalid_length() {
        CscMatrix::<f64>::new(1, 2, vec![0, 1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_rowind() {
        CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![1], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_rowind_values() {
        CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0, 2.0]);
    }
}
