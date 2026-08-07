use crate::{__internal, Element, Vector, api::vector::call};

macro_rules! impl_swizzle_method {
    ($name:ident => [$index:literal]) => {
        // TODO(api-cleanup): Decide whether x(), y(), z(), and w() should return T instead of
        // Vector<T, 1> before restoring one-lane swizzles.
    };
    ($name:ident => [$index0:literal, $index1:literal]) => {
        #[doc = concat!("Returns the lanes `", stringify!($name), "`.")]
        #[inline]
        pub fn $name(self) -> Vector<T, 2> {
            call!(Vector(<T, D>::swizzle2::<$index0, $index1>(self.storage)))
        }
    };
    ($name:ident => [$index0:literal, $index1:literal, $index2:literal]) => {
        #[doc = concat!("Returns the lanes `", stringify!($name), "`.")]
        #[inline]
        pub fn $name(self) -> Vector<T, 3> {
            call!(Vector(
                <T, D>::swizzle3::<$index0, $index1, $index2>(self.storage)
            ))
        }
    };
    (
        $name:ident => [
            $index0:literal,
            $index1:literal,
            $index2:literal,
            $index3:literal
        ]
    ) => {
        #[doc = concat!("Returns the lanes `", stringify!($name), "`.")]
        #[inline]
        pub fn $name(self) -> Vector<T, 4> {
            call!(Vector(
                <T, D>::swizzle4::<$index0, $index1, $index2, $index3>(self.storage)
            ))
        }
    };
}

macro_rules! impl_swizzles {
    ($min:literal; {$($name:ident => [$($index:literal),+],)*}) => {
        #[cfg_attr(not(doc), doc(hidden))]
        impl<T: Element<D>, const D: usize> Vector<T, D>
        where
            __internal::Dimension<D>: __internal::AtLeast<$min>,
        {
            $(impl_swizzle_method!($name => [$($index),+]);)*
        }
    };
}

#[cfg_attr(not(doc), doc(hidden))]
impl<T: Element<D>, const D: usize> Vector<T, D>
where
    __internal::Dimension<D>: __internal::AtLeast<1>,
{
    // TODO(api-cleanup): Decide whether x(), y(), z(), and w() should return T instead of
    // Vector<T, 1> before restoring one-lane swizzles.
    /// Returns the lanes `xx`.
    #[inline]
    pub fn xx(self) -> Vector<T, 2> { Vector::<T, 2>::splat(self.to_array()[0]) }
    /// Returns the lanes `xxx`.
    #[inline]
    pub fn xxx(self) -> Vector<T, 3> { Vector::<T, 3>::splat(self.to_array()[0]) }
    /// Returns the lanes `xxxx`.
    #[inline]
    pub fn xxxx(self) -> Vector<T, 4> { Vector::<T, 4>::splat(self.to_array()[0]) }
}

// build.rs generates swizzle_impls.rs in OUT_DIR at build time.
include!(concat!(env!("OUT_DIR"), "/swizzle_impls.rs"));
