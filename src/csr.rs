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

// Unary / Binary operators
mod ops;

// Conversions
mod conv;

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
    /// Creates a new compressed sparse column matrix.
    ///
    /// # Parameters
    ///
    /// - `nrows`: number of rows
    /// - `ncols`: number of columns
    /// - `rowptr`: row pointers of size `nrows + 1`
    /// - `colind`: column indices of matrix entries
    /// - `values`: values of matrix entries
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    /// // Simple 2 by 3 matrix
    /// // | 1 0 0 |
    /// // | 0 2 3 |
    /// let matrix = CsrMatrix::new(
    ///     2, // number of rows
    ///     3, // number of columns
    ///     vec![0, 1, 3], // row pointers of size 4 = number of rows + 1
    ///     vec![0, 1, 2], // column indices of entries
    ///     vec![1.0, 2.0, 3.0], // values of entries
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `nrows < 1`,
    /// - `ncols < 1`,
    /// - `rowptr.len() != nrows + 1`,
    /// - `rowptr[0] != 0`,
    /// - `colind.len() != rowptr[nrows]`
    /// - `values.len() != rowptr[nrows]`
    /// - `rowptr` is unsorted
    /// - `colind` is not column sorted
    /// - `colind` has invalid column index ∉ `0..cols`
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
        assert_eq!(rowptr[0], 0);
        assert_eq!(colind.len(), rowptr[nrows]);
        assert_eq!(values.len(), rowptr[nrows]);
        assert!(rowptr.windows(2).all(|ptr| ptr[0] <= ptr[1]));
        assert!(colind.iter().all(|col| (0..ncols).contains(col)));
        for row in 0..nrows {
            assert!(colind[rowptr[row]..rowptr[row + 1]]
                .windows(2)
                .all(|cols| cols[0] < cols[1]));
        }
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

    /// Returns the matrix transpose.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CsrMatrix;
    ///
    /// let mut matrix = CsrMatrix::<f64>::new(2, 2, vec![0, 2, 3], vec![0, 1, 1], vec![1.0, 2.0, 3.0]);
    /// let transpose = matrix.transpose();
    /// assert_eq!(transpose.rowptr(), &[0, 1, 3]);
    /// assert_eq!(transpose.colind(), &[0, 0, 1]);
    /// assert_eq!(transpose.values(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn transpose(&self) -> Self {
        let nrows = self.nrows();
        let ncols = self.ncols();
        let nz = self.nnz();
        let rowptr = self.rowptr();
        let colind = self.colind();
        let rowval = self.values();

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

        // Construct matrix
        Self {
            nrows: ncols,
            ncols: nrows,
            rowptr: colptr,
            colind: rowind,
            values: colval,
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
    fn new_unsorted_colind() {
        CsrMatrix::<f64>::new(2, 1, vec![0, 2, 2], vec![1, 0], vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic]
    fn new_invalid_rowind_values() {
        CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0, 2.0]);
    }
}
