// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Verifies: `if (cond) <stmt>;` (no braces, no `return`) is rejected.
// The two recognised forms are `if (cond) return;` and `if (cond) { ... }`;
// single-statement bodies must be wrapped in braces.

use fourierd3_engine::cuda;
use fourierd3_engine::ir::stmt::Stmt;

fn main() {
    let mut v: Vec<Stmt> = Vec::new();
    cuda! { v =>
        if (idx >= 1024) out_0[idx] = 0;
    }
}
