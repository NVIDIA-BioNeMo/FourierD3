// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The marker a compile error carries when no kernel of the requested shape
//! can run on the device at all, as opposed to a compile failure the caller
//! could retry differently.

pub(crate) const INFEASIBLE_PREFIX: &str = "INFEASIBLE: ";

pub(crate) fn infeasible(msg: impl std::fmt::Display) -> String {
    format!("{INFEASIBLE_PREFIX}{msg}")
}

pub(crate) fn is_infeasible(e: &str) -> bool {
    e.starts_with(INFEASIBLE_PREFIX)
}
