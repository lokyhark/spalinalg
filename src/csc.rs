//! Compressed sparse column format module.

use crate::scalar::Scalar;

/// Compressed sparse column (CSC) format matrix.
#[derive(Debug)]
pub struct CscMatrix<T: Scalar> {
    nrows: usize,
    ncols: usize,
    colptr: Vec<usize>,
    rowind: Vec<usize>,
    values: Vec<T>,
}

// Unary / Binary operators
mod ops;

// Conversions
mod conv;

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

impl<T: Scalar> CscMatrix<T> {
    /// Creates a new compressed sparse column matrix.
    ///
    /// # Parameters
    ///
    /// - `nrows`: number of rows
    /// - `ncols`: number of columns
    /// - `colptr`: column pointers of size `ncols + 1`
    /// - `rowind`: row indices of matrix entries
    /// - `values`: values of matrix entries
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    /// // Simple 2 by 3 matrix
    /// // | 1 0 0 |
    /// // | 0 2 3 |
    /// let matrix = CscMatrix::new(
    ///     2, // number of rows
    ///     3, // number of columns
    ///     vec![0, 1, 2, 3], // column pointers of size 4 = number of columns + 1
    ///     vec![0, 1, 1], // row indices of entries
    ///     vec![1.0, 2.0, 3.0], // values of entries
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `nrows < 1`,
    /// - `ncols < 1`,
    /// - `colptr.len() != ncols + 1`,
    /// - `colptr[0] != 0`,
    /// - `rowind.len() != colptr[ncols]`
    /// - `values.len() != colptr[ncols]`
    /// - `colptr` is unsorted
    /// - `rowind` is not column sorted
    /// - `rowind` has invalid column index ∉ `0..nrows`
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
        assert_eq!(colptr[0], 0);
        assert_eq!(rowind.len(), colptr[ncols]);
        assert_eq!(values.len(), colptr[ncols]);
        assert!(colptr.windows(2).all(|ptr| ptr[0] <= ptr[1]));
        assert!(rowind.iter().all(|row| (0..nrows).contains(row)));
        for col in 0..ncols {
            assert!(rowind[colptr[col]..colptr[col + 1]]
                .windows(2)
                .all(|rows| rows[0] < rows[1]));
        }
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

    /// Returns the matrix transpose.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CscMatrix;
    ///
    /// let mut matrix = CscMatrix::<f64>::new(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![1.0, 2.0, 3.0]);
    /// let transpose = matrix.transpose();
    /// assert_eq!(transpose.colptr(), &[0, 2, 3]);
    /// assert_eq!(transpose.rowind(), &[0, 1, 1]);
    /// assert_eq!(transpose.values(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn transpose(&self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let nz = self.nnz();
        let rowind = self.rowind();
        let colptr = self.colptr();
        let colval = self.values();

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

        // Construct matrix
        Self {
            nrows: ncols,
            ncols: nrows,
            colptr: rowptr,
            rowind: colind,
            values: rowval,
        }
    }
}

impl<T: Scalar> IntoIterator for CscMatrix<T> {
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
    fn new_unsorted_rowind() {
        CscMatrix::<f64>::new(1, 2, vec![0, 2, 2], vec![1, 0], vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_rowind_values() {
        CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0, 2.0]);
    }
}
