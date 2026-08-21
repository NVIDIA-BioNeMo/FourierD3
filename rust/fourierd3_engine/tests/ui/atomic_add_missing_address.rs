// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Verifies: atomicAdd's first argument must be `&<lvalue>`.

use fourierd3_engine::cuda;
use fourierd3_engine::ir::stmt::Stmt;

fn main() {
    let mut v: Vec<Stmt> = Vec::new();
    cuda! { v =>
        atomicAdd(out_0[i], val);
    }
}
