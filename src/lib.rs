//! Sparse Linear Algebra Library.
//!
//! The sparse linear algebra library provides convenient matrix format including :
//! - [`CooMatrix`]: Coordinate format matrix intended for incremental matrix construction with duplicates
//! - [`DokMatrix`]: Dictionnary of key format matrix intended for incremental matrix construction without duplicates
//! - [`CsrMatrix`] / [`CscMatrix`]: Compressed sparse matrix intended for standard matrix operations

#![allow(clippy::needless_range_loop)]

pub mod coo;
pub mod csc;
pub mod csr;
pub mod dok;
pub mod scalar;

pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use dok::DokMatrix;
