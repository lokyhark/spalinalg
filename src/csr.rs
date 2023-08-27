//! Compressed sparse row format module.

use crate::{scalar::Scalar, CooMatrix, CscMatrix, DokMatrix};

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

impl<T: Scalar> From<CooMatrix<T>> for CsrMatrix<T> {
    fn from(coo: CooMatrix<T>) -> Self {
        let nrows = coo.nrows();
        let ncols = coo.ncols();
        let len = coo.length();

        // Count number of entries in each row
        let mut vec = vec![0; ncols];
        for (_, col, _) in coo.iter() {
            vec[col] += 1;
        }

        // Construct col pointers
        let mut colptr = Vec::with_capacity(ncols + 1);
        let mut sum = 0;
        colptr.push(0);
        for value in &mut vec {
            sum += *value;
            colptr.push(sum);
        }

        // Construct compressed row form (colptr, rowind, colval)
        let mut vec = colptr[..ncols].to_vec();
        let mut rowind = vec![0; len];
        let mut colval = vec![T::zero(); len];
        for (row, col, val) in coo.into_iter() {
            let ptr = &mut vec[col];
            rowind[*ptr] = row;
            colval[*ptr] = val;
            *ptr += 1
        }

        // Sum up duplicates
        let mut vec = vec![None; nrows];
        let mut nz = 0;
        for col in 0..ncols {
            let start = nz;
            for ptr in colptr[col]..colptr[col + 1] {
                let row = rowind[ptr];
                match vec[row] {
                    Some(prev) if prev >= start => {
                        let val = colval[ptr];
                        colval[prev] += val;
                    }
                    _ => {
                        vec[row] = Some(nz);
                        rowind[nz] = row;
                        colval[nz] = colval[ptr];
                        nz += 1;
                    }
                }
            }
            colptr[col] = start;
        }
        colptr[ncols] = nz;

        // Drop numerically zero entries
        let mut nz = 0;
        for col in 0..ncols {
            let start = std::mem::replace(&mut colptr[col], nz);
            for ptr in start..colptr[col + 1] {
                if colval[ptr] != T::zero() {
                    let ind = rowind[ptr];
                    rowind[nz] = ind;
                    let val = colval[ptr];
                    colval[nz] = val;
                    nz += 1;
                }
            }
        }
        colptr[ncols] = nz;

        // Count number of entries in each row
        let mut vec = vec![0; nrows];
        for col in 0..ncols {
            for row in rowind.iter().take(colptr[col + 1]).skip(colptr[col]) {
                vec[*row] += 1;
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

        // Construct CsrMatrix
        CsrMatrix {
            nrows,
            ncols,
            rowptr,
            colind,
            values: rowval,
        }
    }
}

impl<T: Scalar> From<CscMatrix<T>> for CsrMatrix<T> {
    fn from(csc: CscMatrix<T>) -> Self {
        let nrows = csc.nrows();
        let ncols = csc.ncols();
        let colptr = csc.colptr();
        let rowind = csc.rowind();
        let colval = csc.values();
        let nz = csc.nnz();

        // Count number of entries in each row
        let mut vec = vec![0; nrows];
        for col in 0..ncols {
            for row in rowind.iter().take(colptr[col + 1]).skip(colptr[col]) {
                vec[*row] += 1;
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

        // Construct CsrMatrix
        CsrMatrix {
            nrows,
            ncols,
            rowptr,
            colind,
            values: rowval,
        }
    }
}

impl<T: Scalar> From<DokMatrix<T>> for CsrMatrix<T> {
    fn from(dok: DokMatrix<T>) -> Self {
        let nrows = dok.nrows();
        let ncols = dok.ncols();
        let nz = dok.length();

        // Count number of entries in each row
        let mut vec = vec![0; ncols];
        for (_, col, _) in dok.iter() {
            vec[col] += 1;
        }

        // Construct col pointers
        let mut colptr = Vec::with_capacity(ncols + 1);
        let mut sum = 0;
        colptr.push(0);
        for value in &mut vec {
            sum += *value;
            colptr.push(sum);
        }

        // Construct compressed row form (colptr, rowind, colval)
        let mut vec = colptr[..ncols].to_vec();
        let mut rowind = vec![0; nz];
        let mut colval = vec![T::zero(); nz];
        for (row, col, val) in dok.into_iter() {
            let ptr = &mut vec[col];
            rowind[*ptr] = row;
            colval[*ptr] = val;
            *ptr += 1
        }

        // Count number of entries in each row
        let mut vec = vec![0; nrows];
        for col in 0..ncols {
            for row in rowind.iter().take(colptr[col + 1]).skip(colptr[col]) {
                vec[*row] += 1;
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

        // Construct CsrMatrix
        CsrMatrix {
            nrows,
            ncols,
            rowptr,
            colind,
            values: rowval,
        }
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
        let csr = CsrMatrix::from(coo);
        assert_eq!(csr.rowptr, [0, 3, 4]);
        assert_eq!(csr.colind, [0, 1, 2, 2]);
        assert_eq!(csr.values, [3.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn from_csc() {
        let csc = CscMatrix::new(
            2,
            3,
            vec![0, 1, 2, 4],
            vec![0, 0, 0, 1],
            vec![3.0, 3.0, 4.0, 5.0],
        );
        let csr = CsrMatrix::from(csc);
        assert_eq!(csr.rowptr, [0, 3, 4]);
        assert_eq!(csr.colind, [0, 1, 2, 2]);
        assert_eq!(csr.values, [3.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn from_dok() {
        let mut dok = DokMatrix::new(2, 3);
        dok.insert(0, 0, 3.0);
        dok.insert(0, 1, 3.0);
        dok.insert(0, 2, 4.0);
        dok.insert(1, 2, 5.0);
        let csr = CsrMatrix::from(dok);
        assert_eq!(csr.rowptr, [0, 3, 4]);
        assert_eq!(csr.colind, [0, 1, 2, 2]);
        assert_eq!(csr.values, [3.0, 3.0, 4.0, 5.0]);
    }
}
