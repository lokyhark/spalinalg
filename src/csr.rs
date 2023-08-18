//! Compressed sparse row format module.

/// Compressed sparse row (CSR) format matrix.
#[derive(Debug)]
pub struct CsrMatrix<T> {
    nrows: usize,
    ncols: usize,
    rowptr: Vec<usize>,
    colind: Vec<usize>,
    values: Vec<T>,
}

impl<T> CsrMatrix<T> {
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
}
