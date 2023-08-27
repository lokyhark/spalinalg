//! Dictionnary of Keys format module.

use std::collections::HashMap;

use crate::{scalar::Scalar, CooMatrix, CscMatrix, CsrMatrix};

/// Dictionnary of Keys (DOK) format sparse matrix.
///
/// # Format
///
/// The dictionnary of keys storage format stores matrix entries by `(row, col)` indices.
///
/// # Properties
///
/// The dictionnary of keys format is intended for incremental matrix constructions. Triplets
/// `(row, col, value)` can be inserted in the dictionnary matrix in any order and duplicates
/// entries are **not** allowed (triplets with same row/column indices).
///
/// The dictionnary of keys format is not intended for standard arithmetic operation and conversion
/// to more efficient format (CSR/CSC) for these operations is recommended.
///
/// The dictionnary of keys format is **not** ordered and allows to access matrix entries by
/// `(row, col)` indices.
///
/// # Methods
///
/// ## Constructors
///
/// - Create an empty matrix [`DokMatrix::new`]
/// - Create an empty matrix and reserve capacity entries [`DokMatrix::with_capacity`]
/// - Create a matrix from entries [`DokMatrix::with_entries`]
/// - Create a matrix from triplets [`DokMatrix::with_triplets`]
///
/// ## Insertion/Removal
///
/// - Insert an entry [`DokMatrix::insert`]
/// - Insert multiple entries [`DokMatrix::extend`]
/// - Clear entries [`DokMatrix::clear`]
///
/// ## Retrieve
///
/// - Get immutable entry [`DokMatrix::get`]
/// - Get mutable entry [`DokMatrix::get_mut`]
///
/// ## Iteration
///
/// - Immutable iterator [`DokMatrix::iter`]
/// - Mutable iterator [`DokMatrix::iter_mut`]
/// - Move iterator [`DokMatrix::into_iter`]
#[derive(Clone, Debug)]
pub struct DokMatrix<T: Scalar> {
    nrows: usize,
    ncols: usize,
    entries: HashMap<(usize, usize), T>,
}

/// Immutable dictionnary of keys matrix entries iterator created by [`DokMatrix::iter`] method.
#[derive(Clone, Debug)]
pub struct Iter<'iter, T> {
    iter: std::collections::hash_map::Iter<'iter, (usize, usize), T>,
}

/// Mutable dictionnary of keys matrix entries iterator created by [`DokMatrix::iter_mut`] method.
#[derive(Debug)]
pub struct IterMut<'iter, T> {
    iter: std::collections::hash_map::IterMut<'iter, (usize, usize), T>,
}

/// Move dictionnary of keys matrix entries iterator created by [`DokMatrix::into_iter`] method.
#[derive(Debug)]
pub struct IntoIter<T> {
    iter: std::collections::hash_map::IntoIter<(usize, usize), T>,
}

impl<T: Scalar> DokMatrix<T> {
    /// Creates a new dictionnary of keys matrix with `nrows` rows and `ncols` columns.
    ///
    /// # Properties
    ///
    /// The created matrix has following properties:
    /// - the matrix is empty (`matrix.length() == 0`)
    /// - the matrix has no capacity (`matrix.capacity() == 0`)
    /// - the matrix will **not** allocate memory before any insert operation
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
    /// use spalinalg::DokMatrix;
    ///
    /// let matrix: DokMatrix<f64> = DokMatrix::new(1, 2);
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
            entries: HashMap::new(),
        }
    }

    /// Creates a new dictionnary of keys matrix with `nrows` rows, `ncols` columns and specified `capacity`.
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
    /// use spalinalg::DokMatrix;
    ///
    /// let matrix: DokMatrix<f64> = DokMatrix::with_capacity(1, 2, 2);
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
            entries: HashMap::with_capacity(capacity),
        }
    }

    /// Creates a new dictionnary of keys matrix with `nrows` rows, `ncols` columns and specified `entries`.
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
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let matrix = DokMatrix::with_entries(2, 3, entries);
    /// assert_eq!(matrix.nrows(), 2);
    /// assert_eq!(matrix.ncols(), 3);
    /// assert_eq!(matrix.length(), 2);
    /// assert!(matrix.capacity() >= 2);
    /// assert_eq!(matrix.get(0, 0), Some(&1.0));
    /// assert_eq!(matrix.get(1, 1), Some(&2.0));
    /// ```
    pub fn with_entries<I>(nrows: usize, ncols: usize, entries: I) -> Self
    where
        I: IntoIterator<Item = (usize, usize, T)>,
    {
        assert!(nrows > 0);
        assert!(ncols > 0);
        let entries: HashMap<_, _> = entries.into_iter().map(|(r, c, v)| ((r, c), v)).collect();
        for (row, col) in entries.keys() {
            assert!(*row < nrows);
            assert!(*col < ncols);
        }
        Self {
            nrows,
            ncols,
            entries,
        }
    }

    /// Creates a new dictionnary of keys matrix with `nrows` rows, `ncols` columns and specified triplets.
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
    /// use spalinalg::DokMatrix;
    ///
    /// let rows = [0, 1];
    /// let cols = [1, 0];
    /// let values = [1.0, 2.0];
    /// let matrix = DokMatrix::with_triplets(2, 3, rows, cols, values);
    /// assert_eq!(matrix.nrows(), 2);
    /// assert_eq!(matrix.ncols(), 3);
    /// assert_eq!(matrix.length(), 2);
    /// assert!(matrix.capacity() >= 2);
    /// assert_eq!(matrix.get(0, 1), Some(&1.0));
    /// assert_eq!(matrix.get(1, 0), Some(&2.0));
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
        let mut entries = HashMap::with_capacity(values.len());
        for (idx, value) in values.into_iter().enumerate() {
            entries.insert((rowind[idx], colind[idx]), value);
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
    /// use spalinalg::DokMatrix;
    ///
    /// let matrix = DokMatrix::<f64>::new(1, 2);
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
    /// use spalinalg::DokMatrix;
    ///
    /// let matrix = DokMatrix::<f64>::new(1, 2);
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
    /// use spalinalg::DokMatrix;
    ///
    /// let matrix = DokMatrix::<f64>::new(1, 2);
    /// assert_eq!(matrix.shape(), (1, 2));
    /// ```
    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Returns the number of entries/non-zeros of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::DokMatrix;
    ///
    /// let matrix = DokMatrix::<f64>::new(2, 2);
    /// assert_eq!(matrix.length(), 0);
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let matrix = DokMatrix::with_entries(2, 2, entries);
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
    /// use spalinalg::DokMatrix;
    ///
    /// let matrix = DokMatrix::<f64>::new(2, 2);
    /// assert_eq!(matrix.capacity(), 0);
    ///
    /// let matrix = DokMatrix::<f64>::with_capacity(2, 2, 1);
    /// assert!(matrix.capacity() >= 1);
    /// ```
    pub fn capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Determines if the matrix contains a non zero at `(row, col)`.
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
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let matrix = DokMatrix::with_entries(2, 2, entries);
    /// assert!(matrix.contains(0, 0));
    /// assert!(matrix.contains(1, 1));
    /// assert!(!matrix.contains(1, 0));
    /// assert!(!matrix.contains(0, 1));
    pub fn contains(&self, row: usize, col: usize) -> bool {
        assert!(row < self.nrows);
        assert!(col < self.ncols);
        self.entries.contains_key(&(row, col))
    }

    /// Returns a reference to an entry of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let matrix = DokMatrix::with_entries(2, 2, entries);
    /// assert_eq!(matrix.get(0, 0), Some(&1.0));
    /// assert_eq!(matrix.get(1, 1), Some(&2.0));
    /// assert!(matrix.get(1, 0).is_none());
    /// assert!(matrix.get(0, 1).is_none());
    /// ```
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        assert!(row < self.nrows);
        assert!(col < self.ncols);
        self.entries.get(&(row, col))
    }

    /// Returns a mutable reference to an entry of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let mut matrix = DokMatrix::with_entries(2, 2, entries);
    /// assert_eq!(matrix.get_mut(0, 0), Some(&mut 1.0));
    /// assert_eq!(matrix.get_mut(1, 1), Some(&mut 2.0));
    /// assert!(matrix.get(1, 0).is_none());
    /// assert!(matrix.get(0, 1).is_none());
    /// ```
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        assert!(row < self.nrows);
        assert!(col < self.ncols);
        self.entries.get_mut(&(row, col))
    }

    /// Insert an entry into the matrix.
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
    /// use spalinalg::DokMatrix;
    ///
    /// let mut matrix = DokMatrix::<f64>::new(1, 1);
    /// matrix.insert(0, 0, 1.0);
    /// assert_eq!(matrix.get(0, 0), Some(&1.0));
    /// ```
    pub fn insert(&mut self, row: usize, col: usize, value: T) -> Option<T> {
        assert!(row < self.nrows);
        assert!(col < self.ncols);
        self.entries.insert((row, col), value)
    }

    /// Clear all entries from the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let mut matrix = DokMatrix::with_entries(2, 2, entries);
    /// assert_eq!(matrix.length(), 2);
    /// matrix.clear();
    /// assert_eq!(matrix.length(), 0)
    /// ```
    pub fn clear(&mut self) {
        self.entries.clear()
    }

    /// Returns an iterator over matrix entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    /// ];
    /// let matrix = DokMatrix::with_entries(1, 1, entries);
    /// let mut iter = matrix.iter();
    /// assert_eq!(iter.next(), Some((0, 0, &1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter {
            iter: self.entries.iter(),
        }
    }

    /// Returns a mutable iterator over matrix entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    /// ];
    /// let mut matrix = DokMatrix::with_entries(1, 1, entries);
    /// let mut iter = matrix.iter_mut();
    /// assert_eq!(iter.next(), Some((0, 0, &mut 1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            iter: self.entries.iter_mut(),
        }
    }
}

impl<T: Scalar> Extend<(usize, usize, T)> for DokMatrix<T> {
    /// Extends dictionnary of keys matrix entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let mut matrix = DokMatrix::new(2, 2);
    /// matrix.extend(entries);
    /// assert_eq!(matrix.get(0, 0), Some(&1.0));
    /// assert_eq!(matrix.get(1, 1), Some(&2.0));
    /// ```
    fn extend<I: IntoIterator<Item = (usize, usize, T)>>(&mut self, iter: I) {
        let entries: Vec<_> = iter.into_iter().collect();
        for (row, col, _) in &entries {
            assert!(*row < self.nrows);
            assert!(*col < self.ncols);
        }
        self.entries
            .extend(entries.into_iter().map(|(r, c, v)| ((r, c), v)));
    }
}

impl<T: Scalar> IntoIterator for DokMatrix<T> {
    type Item = (usize, usize, T);

    type IntoIter = IntoIter<T>;

    /// Turns dictionnary of keys matrix into iterator over entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::DokMatrix;
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    /// ];
    /// let mut matrix = DokMatrix::with_entries(1, 1, entries);
    /// let mut iter = matrix.into_iter();
    /// assert_eq!(iter.next(), Some((0, 0, 1.0)));
    /// assert!(iter.next().is_none());
    /// ```
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            iter: self.entries.into_iter(),
        }
    }
}

impl<'iter, T> Iterator for Iter<'iter, T> {
    type Item = (usize, usize, &'iter T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|((r, c), v)| (*r, *c, v))
    }
}

impl<'iter, T> Iterator for IterMut<'iter, T> {
    type Item = (usize, usize, &'iter mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|((r, c), v)| (*r, *c, v))
    }
}

impl<T: Scalar> Iterator for IntoIter<T> {
    type Item = (usize, usize, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|((r, c), v)| (r, c, v))
    }
}

impl<T: Scalar> From<CooMatrix<T>> for DokMatrix<T> {
    /// Conversion from COO format to DOK format.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::{CooMatrix, DokMatrix};
    ///
    /// let entries = vec![
    ///     (0, 0, 1.0),
    ///     (1, 1, 2.0),
    /// ];
    /// let coo = CooMatrix::with_entries(2, 3, entries);
    /// let dok = DokMatrix::from(coo);
    /// ```
    fn from(coo: CooMatrix<T>) -> Self {
        let nrows = coo.nrows();
        let ncols = coo.ncols();
        let mut map = HashMap::with_capacity(coo.length());
        for (row, col, value) in coo.into_iter() {
            *map.entry((row, col)).or_default() += value;
        }
        DokMatrix {
            nrows,
            ncols,
            entries: map,
        }
    }
}

impl<T: Scalar> From<CscMatrix<T>> for DokMatrix<T> {
    /// Conversion from CSC format to DOK format.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::{CscMatrix, DokMatrix};
    ///
    /// let csc = CscMatrix::<f64>::new(1, 2, vec![0, 1, 1], vec![0], vec![1.0]);
    /// let dok = DokMatrix::from(csc);
    /// ```
    fn from(csc: CscMatrix<T>) -> Self {
        DokMatrix {
            nrows: csc.nrows(),
            ncols: csc.ncols(),
            entries: csc.into_iter().map(|(r, c, v)| ((r, c), v)).collect(),
        }
    }
}

impl<T: Scalar> From<CsrMatrix<T>> for DokMatrix<T> {
    /// Conversion from CSR format to DOK format.
    ///
    /// # Examples
    ///
    /// ```
    /// use spalinalg::{CsrMatrix, DokMatrix};
    ///
    /// let csr = CsrMatrix::<f64>::new(2, 1, vec![0, 1, 1], vec![0], vec![1.0]);
    /// let dok = DokMatrix::from(csr);
    /// ```
    fn from(csr: CsrMatrix<T>) -> Self {
        DokMatrix {
            nrows: csr.nrows(),
            ncols: csr.ncols(),
            entries: csr.into_iter().map(|(r, c, v)| ((r, c), v)).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let matrix: DokMatrix<f64> = DokMatrix::new(1, 2);
        assert_eq!(matrix.nrows(), 1);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix.length(), 0);
        assert_eq!(matrix.capacity(), 0);
    }

    #[test]
    #[should_panic]
    fn new_invalid_nrows() {
        DokMatrix::<f64>::new(0, 1);
    }

    #[test]
    #[should_panic]
    fn new_invalid_ncols() {
        DokMatrix::<f64>::new(1, 0);
    }

    #[test]
    fn with_capacity() {
        let matrix: DokMatrix<f64> = DokMatrix::with_capacity(1, 2, 4);
        assert_eq!(matrix.nrows(), 1);
        assert_eq!(matrix.ncols(), 2);
        assert_eq!(matrix.length(), 0);
        assert!(matrix.capacity() >= 4);
    }

    #[test]
    #[should_panic]
    fn with_capacity_invalid_nrows() {
        DokMatrix::<f64>::with_capacity(0, 1, 1);
    }

    #[test]
    #[should_panic]
    fn with_capacity_invalid_ncols() {
        DokMatrix::<f64>::with_capacity(0, 1, 1);
    }

    #[test]
    fn with_entries() {
        let entries = vec![(0, 0, 1.0), (1, 0, 2.0), (0, 2, 3.0)];
        let matrix = DokMatrix::with_entries(2, 3, entries);
        assert_eq!(matrix.length(), 3);
        assert!(matrix.capacity() >= 3);
        assert_eq!(matrix.get(0, 0), Some(&1.0));
        assert_eq!(matrix.get(1, 0), Some(&2.0));
        assert_eq!(matrix.get(0, 2), Some(&3.0));
    }

    #[test]
    #[should_panic]
    fn with_entries_invalid_nrows() {
        DokMatrix::<f64>::with_entries(0, 1, vec![]);
    }

    #[test]
    #[should_panic]
    fn with_entries_invalid_ncols() {
        DokMatrix::<f64>::with_entries(1, 0, vec![]);
    }

    #[test]
    #[should_panic]
    fn with_entries_invalid_row() {
        DokMatrix::<f64>::with_entries(1, 1, vec![(1, 0, 1.0)]);
    }

    #[test]
    #[should_panic]
    fn with_entries_invalid_col() {
        DokMatrix::<f64>::with_entries(1, 1, vec![(0, 1, 1.0)]);
    }

    #[test]
    fn with_triplets() {
        let rowind = vec![0, 1];
        let colind = vec![1, 0];
        let values = vec![-1.0, 1.0];
        let matrix = DokMatrix::with_triplets(2, 2, rowind, colind, values);
        assert_eq!(matrix.length(), 2);
        assert!(matrix.capacity() >= 2);
        assert_eq!(matrix.get(0, 1), Some(&-1.0));
        assert_eq!(matrix.get(1, 0), Some(&1.0));
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_nrows() {
        DokMatrix::<f64>::with_triplets(0, 1, vec![], vec![], vec![]);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_ncols() {
        DokMatrix::<f64>::with_triplets(1, 0, vec![], vec![], vec![]);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_triplets_rowind_length() {
        DokMatrix::<f64>::with_triplets(1, 1, vec![], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_triplets_colind_length() {
        DokMatrix::<f64>::with_triplets(1, 1, vec![0], vec![], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_triplets_values_length() {
        DokMatrix::<f64>::with_triplets(1, 1, vec![0], vec![0], vec![]);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_row() {
        DokMatrix::<f64>::with_triplets(1, 1, vec![1], vec![0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn with_triplets_invalid_col() {
        DokMatrix::<f64>::with_triplets(1, 1, vec![0], vec![1], vec![1.0]);
    }

    #[test]
    fn nrows() {
        let matrix: DokMatrix<f64> = DokMatrix::new(1, 2);
        assert_eq!(matrix.nrows(), 1);
    }

    #[test]
    fn ncols() {
        let matrix: DokMatrix<f64> = DokMatrix::new(1, 2);
        assert_eq!(matrix.ncols(), 2);
    }

    #[test]
    fn shape() {
        let matrix: DokMatrix<f64> = DokMatrix::new(1, 2);
        assert_eq!(matrix.shape(), (1, 2));
    }

    #[test]
    fn length() {
        let mut matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        assert_eq!(matrix.length(), 0);
        matrix.insert(0, 0, 1.0);
        assert_eq!(matrix.length(), 1);
    }

    #[test]
    fn capacity() {
        let mut matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        assert_eq!(matrix.capacity(), 0);
        matrix.insert(0, 0, 1.0);
        assert!(matrix.capacity() >= 1);
    }

    #[test]
    fn contains() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = DokMatrix::with_entries(1, 2, entries);
        assert!(matrix.contains(0, 0));
        assert!(!matrix.contains(0, 1));
    }

    #[test]
    #[should_panic]
    fn contains_invalid_row() {
        let matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        matrix.contains(1, 0);
    }

    #[test]
    #[should_panic]
    fn contains_invalid_col() {
        let matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        matrix.contains(0, 1);
    }

    #[test]
    fn get() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = DokMatrix::with_entries(2, 2, entries);
        assert_eq!(matrix.get(0, 0), Some(&1.0));
        assert!(matrix.get(0, 1).is_none());
    }

    #[test]
    #[should_panic]
    fn get_invalid_row() {
        let matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        matrix.get(1, 0);
    }

    #[test]
    #[should_panic]
    fn get_invalid_col() {
        let matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        matrix.get(0, 1);
    }

    #[test]
    fn get_mut() {
        let entries = vec![(0, 0, 1.0)];
        let mut matrix = DokMatrix::with_entries(1, 2, entries);
        assert_eq!(matrix.get_mut(0, 0), Some(&mut 1.0));
        assert!(matrix.get_mut(0, 1).is_none());
    }

    #[test]
    #[should_panic]
    fn get_mut_invalid_row() {
        let mut matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        matrix.get_mut(1, 0);
    }

    #[test]
    #[should_panic]
    fn get_mut_invalid_col() {
        let mut matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        matrix.get_mut(0, 1);
    }

    #[test]
    fn insert() {
        let mut matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        matrix.insert(0, 0, 1.0);
        assert_eq!(matrix.get(0, 0), Some(&1.0));
    }

    #[test]
    #[should_panic]
    fn insert_invalid_row() {
        let mut matrix = DokMatrix::new(1, 1);
        matrix.insert(1, 0, 1.0);
    }

    #[test]
    #[should_panic]
    fn insert_invalid_col() {
        let mut matrix: DokMatrix<f64> = DokMatrix::new(1, 1);
        matrix.insert(0, 1, 1.0);
    }

    #[test]
    fn clear() {
        let entries = vec![(0, 0, 1.0)];
        let mut matrix = DokMatrix::with_entries(1, 1, entries);
        matrix.clear();
        assert_eq!(matrix.length(), 0);
    }

    #[test]
    fn iter() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = DokMatrix::with_entries(1, 1, entries);
        let mut iter = matrix.iter();
        assert_eq!(iter.next(), Some((0, 0, &1.0)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn iter_mut() {
        let entries = vec![(0, 0, 1.0)];
        let mut matrix = DokMatrix::with_entries(1, 1, entries);
        let mut iter = matrix.iter_mut();
        assert_eq!(iter.next(), Some((0, 0, &mut 1.0)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn extend() {
        let entries = vec![(0, 0, 1.0), (1, 0, 2.0), (0, 2, 3.0)];
        let mut matrix = DokMatrix::new(2, 3);
        matrix.extend(entries);
        assert_eq!(matrix.length(), 3);
        assert!(matrix.capacity() >= 3);
        assert_eq!(matrix.get(0, 0), Some(&1.0));
        assert_eq!(matrix.get(1, 0), Some(&2.0));
        assert_eq!(matrix.get(0, 2), Some(&3.0));
    }

    #[test]
    fn into_iter() {
        let entries = vec![(0, 0, 1.0)];
        let matrix = DokMatrix::with_entries(1, 1, entries);
        let mut iter = matrix.into_iter();
        assert_eq!(iter.next(), Some((0, 0, 1.0)));
        assert!(iter.next().is_none());
    }
}
