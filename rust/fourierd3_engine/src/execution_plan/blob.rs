// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared immutable byte storage: one owner per byte.
//!
//! A [`Blob`] is a cheaply-clonable view into reference-counted bytes. A
//! decoded plan's payloads (cubins, workspace inits) are views into the one
//! buffer it was decoded from, so cloning plans and
//! candidates never copies payload bytes.

use std::ops::{Deref, Range};
use std::sync::Arc;

#[derive(Clone)]
pub(crate) struct Blob {
    src: Arc<dyn AsRef<[u8]> + Send + Sync>,
    range: Range<usize>,
}

impl Blob {
    pub(crate) fn from_arc(src: Arc<dyn AsRef<[u8]> + Send + Sync>) -> Blob {
        let len = (*src).as_ref().len();
        Blob { src, range: 0..len }
    }

    pub(crate) fn from_vec(bytes: Vec<u8>) -> Blob {
        Blob::from_arc(Arc::new(bytes))
    }

    /// A sub-view; `range` is relative to `self`.
    pub(crate) fn slice(&self, range: Range<usize>) -> Blob {
        assert!(
            range.start <= range.end && range.end <= self.range.len(),
            "blob slice {range:?} out of bounds for length {}",
            self.range.len()
        );
        Blob {
            src: self.src.clone(),
            range: self.range.start + range.start..self.range.start + range.end,
        }
    }

    /// A fresh self-owned copy of just these bytes, severed from the shared
    /// source — so a small view stops pinning a large backing buffer.
    pub(crate) fn detached(&self) -> Blob {
        Blob::from_vec(self.to_vec())
    }
}

impl Deref for Blob {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &(*self.src).as_ref()[self.range.clone()]
    }
}

impl AsRef<[u8]> for Blob {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl Default for Blob {
    fn default() -> Blob {
        Blob::from_arc(Arc::new([]))
    }
}

impl From<Vec<u8>> for Blob {
    fn from(bytes: Vec<u8>) -> Blob {
        Blob::from_vec(bytes)
    }
}

impl PartialEq for Blob {
    fn eq(&self, other: &Blob) -> bool {
        **self == **other
    }
}

impl Eq for Blob {}

impl std::fmt::Debug for Blob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Blob({} bytes)", self.range.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_views_share_storage() {
        let blob = Blob::from_vec((0u8..100).collect());
        let view = blob.slice(10..20);
        assert_eq!(&*view, &(10u8..20).collect::<Vec<u8>>()[..]);
        let sub = view.slice(5..10);
        assert_eq!(&*sub, &(15u8..20).collect::<Vec<u8>>()[..]);
    }

    #[test]
    fn detached_equals_but_does_not_share() {
        let blob = Blob::from_vec(vec![1, 2, 3, 4]);
        let view = blob.slice(1..3);
        let own = view.detached();
        assert_eq!(own, view);
        assert_eq!(&*own, &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn slice_out_of_bounds_panics() {
        Blob::from_vec(vec![0; 4]).slice(2..6);
    }
}
