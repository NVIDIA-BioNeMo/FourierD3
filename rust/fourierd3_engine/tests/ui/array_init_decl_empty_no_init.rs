// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Verifies: empty brackets without an initializer list are rejected.
// `i32 x[];` is only valid C in an `extern` declaration we don't model.
// To declare an uninitialised array, supply the size: `i32 x[3];`.

use fourierd3_engine::cuda;
use fourierd3_engine::ir::stmt::Stmt;

fn main() {
    let mut v: Vec<Stmt> = Vec::new();
    cuda! { v =>
        i32 x[];
    }
}
