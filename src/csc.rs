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
}
