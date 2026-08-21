// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Verifies: parse errors accumulate. Three independent mistakes in one
// macro invocation must all be reported in a single compile pass — the
// user shouldn't have to fix one, recompile, see the next, fix it,
// recompile, ad nauseam.

use fourierd3_engine::cuda;
use fourierd3_engine::ir::stmt::Stmt;

fn main() {
    let mut v: Vec<Stmt> = Vec::new();
    cuda! { v =>
        if (idx < n) idx = idx + 1;
        x + y;
        atomicAdd(out[i], val);
    }
}
