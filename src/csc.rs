//! Compressed sparse column format module.

use crate::{scalar::Scalar, CooMatrix};

/// Compressed sparse column (CSC) format matrix.
#[derive(Debug)]
pub struct CscMatrix<T: Scalar> {
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

impl<T: Scalar> CscMatrix<T> {
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

impl<T: Scalar> From<CooMatrix<T>> for CscMatrix<T> {
    fn from(coo: CooMatrix<T>) -> Self {
        let nrows = coo.nrows();
        let ncols = coo.ncols();
        let len = coo.length();

        // Count number of entries in each row
        let mut vec = vec![0; nrows];
        for (row, _, _) in coo.iter() {
            vec[row] += 1;
        }

        // Construct row pointers
        let mut rowptr = Vec::with_capacity(nrows + 1);
        let mut sum = 0;
        rowptr.push(0);
        for value in &mut vec {
            sum += *value;
            rowptr.push(sum);
        }

        // Construct compressed row form (rowptr, colind, values)
        let mut vec = rowptr[..nrows].to_vec();
        let mut colind = vec![0; len];
        let mut rowval = vec![T::zero(); len];
        for (row, col, val) in coo.into_iter() {
            let ptr = vec[row];
            vec[row] += 1;
            colind[ptr] = col;
            rowval[ptr] = val;
        }

        // Sum up duplicates
        let mut vec = vec![None; ncols];
        let mut nz = 0;
        for row in 0..nrows {
            let start = nz;
            for ptr in rowptr[row]..rowptr[row + 1] {
                let col = colind[ptr];
                match vec[col] {
                    Some(prev) if prev >= start => {
                        let val = rowval[ptr];
                        rowval[prev] += val;
                    }
                    _ => {
                        vec[col] = Some(nz);
                        colind[nz] = col;
                        rowval[nz] = rowval[ptr];
                        nz += 1;
                    }
                }
            }
            rowptr[row] = start;
        }
        rowptr[nrows] = nz;

        // Drop numerically zero entries
        let mut nz = 0;
        for row in 0..nrows {
            let start = std::mem::replace(&mut rowptr[row], nz);
            for ptr in start..rowptr[row + 1] {
                if rowval[ptr] != T::zero() {
                    let ind = colind[ptr];
                    colind[nz] = ind;
                    let val = rowval[ptr];
                    rowval[nz] = val;
                    nz += 1;
                }
            }
        }
        rowptr[nrows] = nz;

        // Count number of entries in each column
        let mut vec = vec![0; ncols];
        for row in 0..nrows {
            for col in colind.iter().take(rowptr[row + 1]).skip(rowptr[row]) {
                vec[*col] += 1;
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

        // Construct CscMatrix
        CscMatrix {
            nrows,
            ncols,
            colptr,
            rowind,
            values: colval,
        }
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

    #[test]
    fn from_coo() {
        // | 1.0 + 2.0  3.0  4.0 |
        // |    0.0     0.0  5.0 |
        let mut coo = CooMatrix::new(2, 3);
        coo.push(1, 2, 5.0); // |
        coo.push(0, 2, 4.0); // |> unsorted rows
        coo.push(0, 1, 3.0); // |
        coo.push(0, 0, 1.0); // |> unsorted cols
        coo.push(0, 0, 2.0); // duplicate
        coo.push(1, 0, 0.0); // zero entry
        coo.push(1, 1, 1.00); // |
        coo.push(1, 1, -1.0); // |> numerical cancel
        let csc = CscMatrix::from(coo);
        assert_eq!(csc.colptr, [0, 1, 2, 4]);
        assert_eq!(csc.rowind, [0, 0, 0, 1]);
        assert_eq!(csc.values, [3.0, 3.0, 4.0, 5.0]);
    }
}
