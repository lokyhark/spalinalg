//! Scalar module.

use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

pub trait Zero: Sized + Add<Self, Output = Self> {
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0f32
    }
}

impl Zero for f64 {
    fn zero() -> Self {
        0.0f64
    }
}

pub trait One: Sized + Mul<Self, Output = Self> {
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> Self {
        1.0f32
    }
}

impl One for f64 {
    fn one() -> Self {
        1.0f64
    }
}

pub trait Ops<Rhs = Self, Output = Self>:
    Add<Rhs, Output = Output>
    + Sub<Rhs, Output = Output>
    + Mul<Rhs, Output = Output>
    + Div<Rhs, Output = Output>
    + Neg<Output = Output>
    + AddAssign<Rhs>
    + SubAssign<Rhs>
    + MulAssign<Rhs>
    + DivAssign<Rhs>
{
}
impl Ops for f32 {}
impl Ops for f64 {}

pub trait Scalar: Copy + Debug + Default + PartialEq + PartialOrd + Zero + One + Ops {}
impl Scalar for f32 {}
impl Scalar for f64 {}
