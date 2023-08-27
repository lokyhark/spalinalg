//! Sparse Linear Algebra Library.

pub mod coo;
pub mod csc;
pub mod csr;
pub mod dok;

pub use coo::CooMatrix;
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use dok::DokMatrix;
