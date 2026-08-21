// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use parking_lot::Mutex;
use std::collections::HashMap;

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv1a_64(data: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

fn hash128(key: &[u8]) -> u128 {
    let lo = fnv1a_64(key, FNV_OFFSET);
    let hi = fnv1a_64(key, 0x84222325cbf29ce4);
    (u128::from(hi) << 64) | u128::from(lo)
}

pub(crate) struct Cache {
    mem: Mutex<HashMap<u128, Vec<u8>>>,
}

impl Cache {
    pub(crate) fn new() -> Cache {
        Cache {
            mem: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn get_or_insert<E>(
        &self,
        key: &[u8],
        produce: impl FnOnce() -> Result<Vec<u8>, E>,
    ) -> Result<Vec<u8>, E> {
        let h = hash128(key);
        if let Some(hit) = self.mem.lock().get(&h).cloned() {
            return Ok(hit);
        }
        let bytes = produce()?;
        self.mem.lock().insert(h, bytes.clone());
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Never = ();

    fn put(cache: &Cache, key: &[u8], bytes: Vec<u8>) -> Vec<u8> {
        cache.get_or_insert(key, || Ok::<_, Never>(bytes)).unwrap()
    }

    #[test]
    fn fnv1a_known_value() {
        assert_eq!(fnv1a_64(&[], FNV_OFFSET), FNV_OFFSET);
    }

    #[test]
    fn mem_hit_skips_produce() {
        let cache = Cache::new();
        assert_eq!(put(&cache, b"k", vec![1, 2, 3]), vec![1, 2, 3]);
        let got = cache
            .get_or_insert(b"k", || -> Result<Vec<u8>, Never> {
                panic!("produce called on a hit")
            })
            .unwrap();
        assert_eq!(got, vec![1, 2, 3]);
    }
}
