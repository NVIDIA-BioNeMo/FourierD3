// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) struct CodeBuilder {
    buf: String,
    indent_level: usize,
    needs_indent: bool,
}

const INDENT: &str = "    ";

impl Default for CodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeBuilder {
    pub(crate) fn new() -> Self {
        Self {
            buf: String::new(),
            indent_level: 0,
            needs_indent: false,
        }
    }

    pub(crate) fn block<F: FnOnce(&mut Self)>(&mut self, f: F) {
        self.indent_level += 1;
        f(self);
        self.indent_level = self.indent_level.saturating_sub(1);
    }

    pub(crate) fn push_str(&mut self, s: &str) {
        let mut first = true;
        for line in s.split('\n') {
            if !first {
                self.buf.push('\n');
                self.needs_indent = true;
            }
            if !line.is_empty() {
                self.apply_indent();
                self.buf.push_str(line);
            }
            first = false;
        }
    }

    pub(crate) fn newline(&mut self) {
        self.buf.push('\n');
        self.needs_indent = true;
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.buf.into_bytes()
    }

    pub(crate) fn finish_string(self) -> String {
        self.buf
    }

    fn apply_indent(&mut self) {
        if self.needs_indent {
            for _ in 0..self.indent_level {
                self.buf.push_str(INDENT);
            }
            self.needs_indent = false;
        }
    }
}

impl std::fmt::Write for CodeBuilder {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.push_str(s);
        Ok(())
    }
}

#[macro_export]
macro_rules! emit {
    ($cb:expr, $s:literal) => {
        $cb.push_str($s)
    };
}

#[macro_export]
macro_rules! emit_ln {
    ($cb:expr, $s:literal) => {{
        $cb.push_str($s);
        $cb.newline();
    }};
}

#[macro_export]
macro_rules! cb {
    ($cb:expr, $($arg:tt)*) => {{
        use ::std::fmt::Write as _;
        ::std::write!($cb, $($arg)*).unwrap();
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plaintext_round_trip() {
        let mut cb = CodeBuilder::new();
        emit!(cb, "int x = ");
        cb!(cb, "{};", 42);
        cb.newline();
        emit_ln!(cb, "if (x > 0) {");
        cb.block(|cb| emit!(cb, "x = x + 1;"));
        cb.newline();
        emit!(cb, "}");
        assert_eq!(
            cb.finish_string(),
            "int x = 42;\nif (x > 0) {\n    x = x + 1;\n}"
        );
    }

    #[test]
    fn nested_indent_blocks() {
        let mut cb = CodeBuilder::new();
        emit_ln!(cb, "a {");
        cb.block(|cb| {
            emit_ln!(cb, "b {");
            cb.block(|cb| emit_ln!(cb, "c;"));
            emit_ln!(cb, "}");
        });
        emit_ln!(cb, "}");
        assert_eq!(cb.finish_string(), "a {\n    b {\n        c;\n    }\n}\n");
    }

    #[test]
    fn push_str_reindents_multi_line() {
        let mut cb = CodeBuilder::new();
        emit_ln!(cb, "outer {");
        cb.block(|cb| cb.push_str("first;\nsecond;\nthird;"));
        cb.newline();
        emit_ln!(cb, "}");
        assert_eq!(
            cb.finish_string(),
            "outer {\n    first;\n    second;\n    third;\n}\n"
        );
    }

    #[test]
    fn newline_only_marks_indent() {
        let mut cb = CodeBuilder::new();
        emit!(cb, "a;");
        cb.newline();
        cb.block(|cb| emit!(cb, "b;"));
        assert_eq!(cb.finish_string(), "a;\n    b;");
    }
}
