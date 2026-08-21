// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The bare identifiers and multi-character punctuation the `cuda!` grammar
//! recognizes, as `syn` custom-keyword and custom-punctuation types.

pub(crate) mod kw {
    syn::custom_keyword!(atomicAdd);
    syn::custom_keyword!(splice);
    syn::custom_keyword!(unroll);
    #[allow(non_camel_case_types)]
    mod cuda_kw {
        syn::custom_keyword!(__shared__);
        syn::custom_keyword!(__syncthreads);
    }
    pub(crate) use cuda_kw::{__shared__, __syncthreads};
}

pub(crate) mod punct {
    syn::custom_punctuation!(PlusPlus, ++);
}
