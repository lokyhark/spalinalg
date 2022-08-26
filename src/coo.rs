//! Coordinate format module.

/// Coordinate (COO) format sparse matrix.
///
/// # Format
///
/// The coordinate storage format stores matrix entry *triplet* `(row, col, value)` where
/// `row`/`col` are row/column *indices* of the entry and `value` its *scalar value*.
///
/// # Properties
///
/// The coordinate format is intended for incremental matrix constructions. Triplets
/// `(row, col, value)` can be push/pop into/from the coordinate matrix in any order and duplicates
/// entries are allowed (triplets with same row/column indices).
///
/// The coordinate format is not intended for standard arithmetic operation and conversion to more
/// efficient format (CSR/CSC) for these operations is recommended.
///
/// The coordinate format is ordered and allows to access matrix entries by index.
///
/// # Methods
///
/// ## Constructors
///
/// - Create an empty coordinate matrix [`CooMatrix::new`]
/// - Create an empty matrix and reserve capacity entries [`CooMatrix::with_capacity`]
/// - Create a matrix from entries [`CooMatrix::with_entries`]
/// - Create a matrix from triplets [`CooMatrix::with_triplets`]
///
/// ## Insertion/Removal
///
/// - Push an entry [`CooMatrix::push`]
/// - Pop an entry [`CooMatrix::pop`]
/// - Push multiple entries [`CooMatrix::extend`]
/// - Clear entries [`CooMatrix::clear`]
///
/// ## Retrieve
///
/// - Get immutable entry [`CooMatrix::get`]
/// - Get mutable entry [`CooMatrix::get_mut`]
///
/// ## Iteration
///
/// - Immutable iterator [`CooMatrix::iter`]
/// - Mutable iterator [`CooMatrix::iter_mut`]
/// - Move iterator [`CooMatrix::into_iter`]
#[derive(Clone, Debug)]
pub struct CooMatrix<T> {
    nrows: usize,
    ncols: usize,
    entries: Vec<(usize, usize, T)>,
}

/// Immutable coordinate matrix entries iterator created by [`CooMatrix::iter`] method.
#[derive(Clone, Debug)]
pub struct Iter<'iter, T> {
    iter: std::slice::Iter<'iter, (usize, usize, T)>,
}

/// Mutable coordinate matrix entries iterator created by [`CooMatrix::iter_mut`] method.
#[derive(Debug)]
pub struct IterMut<'iter, T> {
    iter: std::slice::IterMut<'iter, (usize, usize, T)>,
}

/// Move coordinate matrix entries iterator created by [`CooMatrix::into_iter`] method.
#[derive(Debug)]
pub struct IntoIter<T> {
    iter: std::vec::IntoIter<(usize, usize, T)>,
}

impl<T> CooMatrix<T> {
    /// Creates a new coordinate matrix with `nrows` rows and `ncols` columns.
    ///
    /// # Properties
    ///
    /// The created matrix has following properties:
    /// - the matrix is empty (`matrix.length() == 0`)
    /// - the matrix has no capacity (`matrix.capacity() == 0`)
    /// - the matrix will **not** allocate memory before any push operation
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `nrows == 0`
    /// - `ncols == 0`
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let matrix: CooMatrix<f64> = CooMatrix::new(1, 2);
    /// assert_eq!(matrix.nrows(), 1);
    /// assert_eq!(matrix.ncols(), 2);
    /// assert_eq!(matrix.length(), 0);
    /// assert_eq!(matrix.capacity(), 0);
    /// ```
    pub fn new(nrows: usize, ncols: usize) -> Self {
        assert!(nrows > 0);
        assert!(ncols > 0);
        Self {
            nrows,
            ncols,
            entries: Vec::new(),
        }
    }

    /// Creates a new coordinate matrix with `nrows` rows, `ncols` columns and specified `capacity`.
    ///
    /// # Properties
    ///
    /// The created matrix has following properties:
    /// - the matrix is empty (`matrix.length() == 0`)
    /// - the matrix capacity is at least `capacity` (`matrix.capacity() >= capacity`)
    /// - the matrix will be able to hold `capacity` entries without reallocating
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `nrows == 0`
    /// - `ncols == 0`
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let matrix: CooMatrix<f64> = CooMatrix::with_capacity(1, 2, 2);
    /// assert_eq!(matrix.nrows(), 1);
    /// assert_eq!(matrix.ncols(), 2);
    /// assert_eq!(matrix.length(), 0);
    /// assert!(matrix.capacity() >= 2);
    /// ```
    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        assert!(nrows > 0);
        assert!(ncols > 0);
        Self {
            nrows,
            ncols,
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Creates a new coordinate matrix with `nrows` rows, `ncols` columns and specified `entries`.
    ///
    /// # Properties
    ///
    /// The created matrix has following properties:
    /// - the matrix is filled with entries (`matrix.length() == entries.len()`)
    /// - the matrix capacity is at least `entries.len()` (`matrix.capacity() >= entries.len()`)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `nrows == 0`
    /// - `ncols == 0`
    /// - for any entry: `row >= nrows` or `col >= ncols`
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let matrix = CooMatrix::with_entries(2, 3, entries);
    /// assert_eq!(matrix.nrows(), 2);
    /// assert_eq!(matrix.ncols(), 3);
    /// assert_eq!(matrix.length(), 2);
    /// assert!(matrix.capacity() >= 2);
    /// assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
    /// assert_eq!(matrix.get(1), Some((&1, &1, &2.0)));
    /// ```
    pub fn with_entries<I>(nrows: usize, ncols: usize, entries: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        assert!(nrows > 0);
        assert!(ncols > 0);
        let entries: Vec<_> = entries.into_iter().collect();
        for (row, col, _) in entries.iter() {
            assert!(*row < nrows);
            assert!(*col < ncols);
        }
        Self {
            nrows,
            ncols,
            entries,
        }
    }

    /// Creates a new coordinate matrix with `nrows` rows, `ncols` columns and specified `triplets`.
    ///
    /// # Properties
    ///
    /// The created matrix has following properties:
    /// - the matrix is filled with triplets (`matrix.length() == values.len()`)
    /// - the matrix capacity is at least `values.len()` (`matrix.capacity() >= values.len()`)
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `nrows == 0`
    /// - `ncols == 0`
    /// - `values.len() != rowind.len()` or `values.len() != colind.len()`
    /// - for any triplet: `row >= nrows` or `col >= ncols`
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let rows = [0, 1];
    /// let cols = [1, 0];
    /// let values = [1.0, 2.0];
    /// let matrix = CooMatrix::with_triplets(2, 3, rows, cols, values);
    /// assert_eq!(matrix.nrows(), 2);
    /// assert_eq!(matrix.ncols(), 3);
    /// assert_eq!(matrix.length(), 2);
    /// assert!(matrix.capacity() >= 2);
    /// assert_eq!(matrix.get(0), Some((&0, &1, &1.0)));
    /// assert_eq!(matrix.get(1), Some((&1, &0, &2.0)));
    /// ```
    pub fn with_triplets<R, C, V>(
        nrows: usize,
        ncols: usize,
        rowind: R,
        colind: C,
        values: V,
    ) -> Self
    where
        R: IntoIterator<Item = usize>,
        C: IntoIterator<Item = usize>,
        V: IntoIterator<Item = T>,
    {
        assert!(nrows > 0);
        assert!(ncols > 0);
        let rowind: Vec<_> = rowind.into_iter().collect();
        let colind: Vec<_> = colind.into_iter().collect();
        let values: Vec<_> = values.into_iter().collect();
        assert!(rowind.len() == values.len());
        assert!(colind.len() == values.len());
        for row in rowind.iter() {
            assert!(*row < nrows);
        }
        for col in colind.iter() {
            assert!(*col < ncols);
        }
        let mut entries = Vec::with_capacity(values.len());
        for (idx, value) in values.into_iter().enumerate() {
            entries.push((rowind[idx], colind[idx], value))
        }
        Self {
            nrows,
            ncols,
            entries,
        }
    }

    /// Returns number of rows of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let matrix = CooMatrix::<f64>::new(1, 2);
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
    /// use spalinalg::CooMatrix;
    ///
    /// let matrix = CooMatrix::<f64>::new(1, 2);
    /// assert_eq!(matrix.ncols(), 2);
    /// ```
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Returns the shape `(nrows, ncols)` of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let matrix = CooMatrix::<f64>::new(1, 2);
    /// assert_eq!(matrix.shape(), (1, 2));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the number of triplets/entries of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let matrix = CooMatrix::<f64>::new(2, 2);
    /// assert_eq!(matrix.length(), 0);
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let matrix = CooMatrix::with_entries(2, 2, entries);
    /// assert_eq!(matrix.length(), 2);
    /// ```
    pub fn length(&self) -> usize {
        self.entries.len()
    }

    /// Returns the capacity of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let matrix = CooMatrix::<f64>::new(2, 2);
    /// assert_eq!(matrix.capacity(), 0);
    ///
    /// let matrix = CooMatrix::<f64>::with_capacity(2, 2, 1);
    /// assert!(matrix.capacity() >= 1);
    /// ```
    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Returns a reference to an entry of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let matrix = CooMatrix::with_entries(2, 2, entries);
    /// assert_eq!(matrix.get(0), Some((&0, &0, &1.0)));
    /// assert_eq!(matrix.get(1), Some((&1, &1, &2.0)));
    /// assert!(matrix.get(2).is_none())
    /// ```
    pub fn get(&self, index: usize) -> Option<(&usize, &usize, &T)> {
        self.entries
            .get(index)
            .map(|(row, col, value)| (row, col, value))
    }

    /// Returns a mutable reference to an entry of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let mut matrix = CooMatrix::with_entries(2, 2, entries);
    /// assert_eq!(matrix.get_mut(0), Some((&0, &0, &mut 1.0)));
    /// assert_eq!(matrix.get_mut(1), Some((&1, &1, &mut 2.0)));
    /// assert!(matrix.get(2).is_none())
    /// ```
    pub fn get_mut(&mut self, index: usize) -> Option<(&usize, &usize, &mut T)> {
        self.entries
            .get_mut(index)
            .map(|(row, col, value)| (&*row, &*col, value))
    }

    /// Push an entry into the matrix.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `row >= self.nrows()`
    /// - `col >= self.nclos()`
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let mut matrix = CooMatrix::<f64>::new(1, 1);
    /// matrix.push(0, 0, 1.0);
    /// assert_eq!(matrix.get(0), Some((&0, &0, &1.0)))
    /// ```
    pub fn push(&mut self, row: usize, col: usize, value: T) {
        assert!(row < self.nrows);
        assert!(col < self.ncols);
        self.entries.push((row, col, value))
    }

    /// Pop an entry from the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    /// ];
    /// let mut matrix = CooMatrix::with_entries(1, 1, entries);
    /// assert_eq!(matrix.pop(), Some((0, 0, 1.0)));
    /// assert!(matrix.pop().is_none());
    pub fn pop(&mut self) -> Option<(usize, usize, T)> {
        self.entries.pop()
    }

    /// Clear all entries from the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::CooMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let mut matrix = CooMatrix::with_entries(2, 2, entries);
    /// assert_eq!(matrix.length(), 2);
    /// matrix.clear();
    /// assert_eq!(matrix.length(), 0)
    /// ```
    pub fn clear(&mut self) {
        self.entries.clear()
    }

    pub fn iter(&self) -> Iter<T> {
        Iter {
            iter: self.entries.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            iter: self.entries.iter_mut(),
        }
    }
}

impl<T> Extend<(usize, usize, T)> for CooMatrix<T> {
    fn extend<I: IntoIterator<Item = (usize, usize, T)>>(&mut self, iter: I) {
        self.entries.extend(iter)
    }
}

impl<T> IntoIterator for CooMatrix<T> {
    type Item = (usize, usize, T);

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.entries.into_iter(),
        }
    }
}

impl<'iter, T> Iterator for Iter<'iter, T> {
    type Item = (&'iter usize, &'iter usize, &'iter T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(r, c, v)| (r, c, v))
    }
}

impl<'iter, T> Iterator for IterMut<'iter, T> {
    type Item = (&'iter usize, &'iter usize, &'iter mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(r, c, v)| (&*r, &*c, v))
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
    fn new() {
        let matrix: CooMatrix<f64> = CooMatrix::new(1, 2);
        assert_eq!(matrix.nrows(), 1);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix.length(), 0);
        assert_eq!(matrix.capacity(), 0);
    }

    #[test]
    #[should_panic]
    fn new_invalid_nrows() {
        CooMatrix::<f64>::new(0, 1);
    }

    #[test]
    #[should_panic]
    fn new_invalid_ncols() {
        CooMatrix::<f64>::new(1, 0);
    }

    #[test]
    fn with_capacity() {
        let matrix: CooMatrix<f64> = CooMatrix::with_capacity(1, 2, 4);
        assert_eq!(matrix.nrows(), 1);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix.length(), 0);
        assert!(matrix.capacity() >= 4);
    }

    #[test]
    #[should_panic]
    fn with_capacity_invalid_nrows() {
        CooMatrix::<f64>::with_capacity(0, 1, 1);
    }

    #[test]
    #[should_panic]
    fn with_capacity_invalid_ncols() {
        CooMatrix::<f64>::with_capacity(0, 1, 1);
    }
}
