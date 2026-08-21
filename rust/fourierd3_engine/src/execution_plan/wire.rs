// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire format: a content-deduplicated blob pool plus a recursive body.
//!
//! All payload bytes (cubins, workspace inits) live once in the BLOBS
//! section, deduplicated by content across the whole Choice tree. The BODY
//! section holds the structure — modules and inits reference pool entries by
//! index, Choice candidates nest recursively. Decoding is zero-copy: every
//! payload becomes a [`Blob`] view into the one source buffer.

use std::collections::HashMap;

use crate::execution_plan::{
    Arg, Blob, BufRef, ExecutionPlan, KernelModule, Node, Op, WorkspaceBuf, WritableBuf,
};

const MAGIC: [u8; 4] = *b"RCPE";
const VERSION: u16 = 7;

const SECTION_BLOBS: u32 = 1;
const SECTION_BODY: u32 = 2;

const PAYLOAD_START: u64 = 12 + 20 * 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum WireError {
    BadMagic { found: [u8; 4] },
    UnsupportedVersion { found: u16 },
    MissingSection { id: u32 },
    BadTag { what: &'static str, tag: u8 },
    BadUtf8,
    Truncated,
    Oversize { what: &'static str, value: u64 },
    BlobIndexOutOfRange { index: u32 },
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::BadMagic { found } => {
                write!(f, "wrong magic {found:?}, expected {MAGIC:?}")
            }
            WireError::UnsupportedVersion { found } => {
                write!(f, "unsupported version {found}, this build reads {VERSION}")
            }
            WireError::MissingSection { id } => {
                write!(f, "required section {id} is missing")
            }
            WireError::BadTag { what, tag } => {
                write!(f, "unknown {what} tag {tag}")
            }
            WireError::BadUtf8 => write!(f, "string field is not valid UTF-8"),
            WireError::Oversize { what, value } => {
                write!(f, "wire field {what} value {value} exceeds u32::MAX")
            }
            WireError::Truncated => write!(f, "buffer is truncated"),
            WireError::BlobIndexOutOfRange { index } => {
                write!(f, "blob pool index {index} is out of range")
            }
        }
    }
}

impl std::error::Error for WireError {}

/// Every wire count/length/index is a `u32`; this is the one place a
/// too-large `usize` turns into a loud [`WireError::Oversize`] instead of a
/// silent wraparound at encode time.
fn checked_u32(value: usize, what: &'static str) -> Result<u32, WireError> {
    u32::try_from(value).map_err(|_| WireError::Oversize {
        what,
        value: value as u64,
    })
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Writer { bytes: Vec::new() }
    }

    fn u8(&mut self, v: u8) {
        self.bytes.push(v);
    }

    fn u16(&mut self, v: u16) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }

    fn blob(&mut self, b: &[u8]) -> Result<(), WireError> {
        self.u32(checked_u32(b.len(), "blob length")?);
        self.bytes.extend_from_slice(b);
        Ok(())
    }

    fn str(&mut self, s: &str) -> Result<(), WireError> {
        self.blob(s.as_bytes())
    }

    fn count(&mut self, n: usize) -> Result<(), WireError> {
        self.u32(checked_u32(n, "count")?);
        Ok(())
    }
}

/// Reads within one section slice; `base` is the slice's absolute offset in
/// the source buffer, so zero-copy views can be addressed in source space.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    base: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8], base: usize) -> Self {
        Cursor {
            bytes,
            pos: 0,
            base,
        }
    }

    fn abs_pos(&self) -> usize {
        self.base + self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], WireError> {
        let end = self.pos.checked_add(n).ok_or(WireError::Truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or(WireError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, WireError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, WireError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, WireError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, WireError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn blob(&mut self) -> Result<&'a [u8], WireError> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    fn str(&mut self) -> Result<String, WireError> {
        let bytes = self.blob()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| WireError::BadUtf8)
    }

    fn count(&mut self) -> Result<usize, WireError> {
        Ok(self.u32()? as usize)
    }
}

/// Payload bytes interned by content: identical cubins or inits anywhere in
/// the Choice tree share one pool entry.
#[derive(Default)]
struct Pool<'a> {
    blobs: Vec<&'a [u8]>,
    index: HashMap<&'a [u8], u32>,
}

impl<'a> Pool<'a> {
    fn intern(&mut self, bytes: &'a [u8]) -> Result<u32, WireError> {
        if let Some(&i) = self.index.get(bytes) {
            return Ok(i);
        }
        let i = checked_u32(self.blobs.len(), "blob pool size")?;
        self.blobs.push(bytes);
        self.index.insert(bytes, i);
        Ok(i)
    }
}

fn write_bufref(w: &mut Writer, buf: &BufRef) -> Result<(), WireError> {
    let (tag, i) = match buf {
        BufRef::Input(i) => (0, *i),
        BufRef::Output(i) => (1, *i),
        BufRef::Workspace(i) => (2, *i),
    };
    w.u8(tag);
    w.u32(checked_u32(i, "bufref index")?);
    Ok(())
}

fn read_bufref(c: &mut Cursor) -> Result<BufRef, WireError> {
    let tag = c.u8()?;
    let i = c.u32()? as usize;
    match tag {
        0 => Ok(BufRef::Input(i)),
        1 => Ok(BufRef::Output(i)),
        2 => Ok(BufRef::Workspace(i)),
        tag => Err(WireError::BadTag {
            what: "bufref",
            tag,
        }),
    }
}

fn write_arg(w: &mut Writer, arg: &Arg) -> Result<(), WireError> {
    write_bufref(w, &arg.buf)?;
    w.u64(arg.offset as u64);
    Ok(())
}

fn read_arg(c: &mut Cursor) -> Result<Arg, WireError> {
    Ok(Arg {
        buf: read_bufref(c)?,
        offset: c.u64()? as usize,
    })
}

fn write_writablebuf(w: &mut Writer, buf: &WritableBuf) -> Result<(), WireError> {
    let (tag, i) = match buf {
        WritableBuf::Output(i) => (0, *i),
        WritableBuf::Workspace(i) => (1, *i),
    };
    w.u8(tag);
    w.u32(checked_u32(i, "writablebuf index")?);
    Ok(())
}

fn read_writablebuf(c: &mut Cursor) -> Result<WritableBuf, WireError> {
    let tag = c.u8()?;
    match tag {
        0 => Ok(WritableBuf::Output(c.u32()? as usize)),
        1 => Ok(WritableBuf::Workspace(c.u32()? as usize)),
        tag => Err(WireError::BadTag {
            what: "writablebuf",
            tag,
        }),
    }
}

fn write_deps(w: &mut Writer, deps: &[usize]) -> Result<(), WireError> {
    w.count(deps.len())?;
    for &d in deps {
        w.u32(checked_u32(d, "deps entry")?);
    }
    Ok(())
}

fn read_deps(c: &mut Cursor) -> Result<Vec<usize>, WireError> {
    let n = c.count()?;
    (0..n).map(|_| Ok(c.u32()? as usize)).collect()
}

fn write_node<'a>(w: &mut Writer, pool: &mut Pool<'a>, node: &'a Node) -> Result<(), WireError> {
    match &node.op {
        Op::KernelLaunch {
            module,
            entry,
            grid,
            block,
            shmem,
            args,
        } => {
            w.u8(0);
            w.u32(checked_u32(*module, "module index")?);
            w.str(entry)?;
            for &g in grid {
                w.u32(g);
            }
            for &b in block {
                w.u32(b);
            }
            w.u32(*shmem);
            w.count(args.len())?;
            for arg in args {
                write_arg(w, arg)?;
            }
        }
        Op::Memset {
            target,
            value,
            nbytes,
        } => {
            w.u8(1);
            write_writablebuf(w, target)?;
            w.u8(*value);
            w.u64(*nbytes as u64);
        }
        Op::Choice {
            candidates,
            input_binding,
            output_binding,
        } => {
            w.u8(3);
            w.count(input_binding.len())?;
            for binding in input_binding {
                write_bufref(w, binding)?;
            }
            w.count(output_binding.len())?;
            for binding in output_binding {
                write_bufref(w, binding)?;
            }
            w.count(candidates.len())?;
            for candidate in candidates {
                write_body(w, pool, candidate)?;
            }
        }
    }
    write_deps(w, &node.deps)
}

fn read_node(c: &mut Cursor, pool: &[Blob]) -> Result<Node, WireError> {
    let kind = c.u8()?;
    let op = match kind {
        0 => {
            let module = c.u32()? as usize;
            let entry = c.str()?;
            let grid = [c.u32()?, c.u32()?, c.u32()?];
            let block = [c.u32()?, c.u32()?, c.u32()?];
            let shmem = c.u32()?;
            let n_args = c.count()?;
            let args = (0..n_args).map(|_| read_arg(c)).collect::<Result<_, _>>()?;
            Op::KernelLaunch {
                module,
                entry,
                grid,
                block,
                shmem,
                args,
            }
        }
        1 => {
            let target = read_writablebuf(c)?;
            let value = c.u8()?;
            let nbytes = c.u64()? as usize;
            Op::Memset {
                target,
                value,
                nbytes,
            }
        }
        3 => {
            let n_in = c.count()?;
            let input_binding = (0..n_in)
                .map(|_| read_bufref(c))
                .collect::<Result<_, _>>()?;
            let n_out = c.count()?;
            let output_binding = (0..n_out)
                .map(|_| read_bufref(c))
                .collect::<Result<_, _>>()?;
            let n_candidates = c.count()?;
            let candidates = (0..n_candidates)
                .map(|_| read_body(c, pool))
                .collect::<Result<_, _>>()?;
            Op::Choice {
                candidates,
                input_binding,
                output_binding,
            }
        }
        kind => {
            return Err(WireError::BadTag {
                what: "node kind",
                tag: kind,
            });
        }
    };
    let deps = read_deps(c)?;
    Ok(Node { op, deps })
}

fn write_body<'a>(
    w: &mut Writer,
    pool: &mut Pool<'a>,
    plan: &'a ExecutionPlan,
) -> Result<(), WireError> {
    w.count(plan.modules.len())?;
    for m in &plan.modules {
        w.u32(pool.intern(&m.cubin)?);
    }
    w.count(plan.workspace.len())?;
    for ws in &plan.workspace {
        w.u64(ws.nbytes as u64);
        match &ws.init {
            Some(init) => {
                w.u8(1);
                w.u32(pool.intern(init)?);
            }
            None => w.u8(0),
        }
    }
    w.count(plan.nodes.len())?;
    for node in &plan.nodes {
        write_node(w, pool, node)?;
    }
    Ok(())
}

fn pool_blob(c: &mut Cursor, pool: &[Blob]) -> Result<Blob, WireError> {
    let index = c.u32()?;
    pool.get(index as usize)
        .cloned()
        .ok_or(WireError::BlobIndexOutOfRange { index })
}

fn read_body(c: &mut Cursor, pool: &[Blob]) -> Result<ExecutionPlan, WireError> {
    let n_modules = c.count()?;
    let modules = (0..n_modules)
        .map(|_| {
            Ok(KernelModule {
                cubin: pool_blob(c, pool)?,
            })
        })
        .collect::<Result<_, _>>()?;
    let n_workspace = c.count()?;
    let workspace = (0..n_workspace)
        .map(|_| {
            let nbytes = c.u64()? as usize;
            let init = if c.u8()? == 1 {
                Some(pool_blob(c, pool)?)
            } else {
                None
            };
            Ok(WorkspaceBuf { nbytes, init })
        })
        .collect::<Result<_, _>>()?;
    let n_nodes = c.count()?;
    let nodes = (0..n_nodes)
        .map(|_| read_node(c, pool))
        .collect::<Result<_, _>>()?;
    Ok(ExecutionPlan {
        modules,
        workspace,
        nodes,
    })
}

pub(crate) fn serialize(plan: &ExecutionPlan) -> Result<Vec<u8>, WireError> {
    let mut pool = Pool::default();
    let mut body = Writer::new();
    write_body(&mut body, &mut pool, plan)?;

    let blobs_len = 4 + pool.blobs.iter().map(|b| 4 + b.len() as u64).sum::<u64>();

    let mut out = Writer::new();
    out.bytes
        .reserve(PAYLOAD_START as usize + blobs_len as usize + body.bytes.len());
    out.bytes.extend_from_slice(&MAGIC);
    out.u16(VERSION);
    out.u16(0); // reserved
    out.u32(2); // section count

    out.u32(SECTION_BLOBS);
    out.u64(PAYLOAD_START);
    out.u64(blobs_len);
    out.u32(SECTION_BODY);
    out.u64(PAYLOAD_START + blobs_len);
    out.u64(body.bytes.len() as u64);

    out.count(pool.blobs.len())?;
    for b in &pool.blobs {
        out.blob(b)?;
    }
    out.bytes.extend_from_slice(&body.bytes);
    Ok(out.bytes)
}

#[cfg(test)]
pub(crate) fn deserialize(bytes: &[u8]) -> Result<ExecutionPlan, WireError> {
    deserialize_shared(Blob::from_vec(bytes.to_vec()))
}

struct SectionEntry {
    id: u32,
    offset: u64,
    length: u64,
}

fn section(bytes: &[u8], table: &[SectionEntry], id: u32) -> Result<(usize, usize), WireError> {
    let entry = table
        .iter()
        .find(|e| e.id == id)
        .ok_or(WireError::MissingSection { id })?;
    let start = entry.offset as usize;
    let end = start
        .checked_add(entry.length as usize)
        .ok_or(WireError::Truncated)?;
    if end > bytes.len() {
        return Err(WireError::Truncated);
    }
    Ok((start, end))
}

/// Zero-copy decode: every payload blob in the result is a view into
/// `source`, which stays alive as long as any view does.
pub(crate) fn deserialize_shared(source: Blob) -> Result<ExecutionPlan, WireError> {
    let bytes: &[u8] = &source;
    let mut c = Cursor::new(bytes, 0);

    let magic: [u8; 4] = c.take(4)?.try_into().expect("take(4) yields four bytes");
    if magic != MAGIC {
        return Err(WireError::BadMagic { found: magic });
    }
    let version = c.u16()?;
    if version != VERSION {
        return Err(WireError::UnsupportedVersion { found: version });
    }
    let _reserved = c.u16()?;
    let section_count = c.u32()? as usize;

    let mut table = Vec::with_capacity(section_count);
    for _ in 0..section_count {
        let id = c.u32()?;
        let offset = c.u64()?;
        let length = c.u64()?;
        table.push(SectionEntry { id, offset, length });
    }

    let (blobs_start, blobs_end) = section(bytes, &table, SECTION_BLOBS)?;
    let mut c = Cursor::new(&bytes[blobs_start..blobs_end], blobs_start);
    let n_blobs = c.count()?;
    let mut pool = Vec::with_capacity(n_blobs);
    for _ in 0..n_blobs {
        let len = c.u32()? as usize;
        let start = c.abs_pos();
        c.take(len)?;
        pool.push(source.slice(start..start + len));
    }

    let (body_start, body_end) = section(bytes, &table, SECTION_BODY)?;
    let mut c = Cursor::new(&bytes[body_start..body_end], body_start);
    read_body(&mut c, &pool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    const GOLDEN: &[u8] = &[
        // header
        0x52, 0x43, 0x50, 0x45,                         // magic "RCPE"
        0x07, 0x00,                                     // version = 7
        0x00, 0x00,                                     // reserved = 0
        0x02, 0x00, 0x00, 0x00,                         // section_count = 2
        // section table
        0x01, 0x00, 0x00, 0x00,                         // section[0].id = 1 (BLOBS)
        0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // section[0].offset = 52
        0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // section[0].length = 20
        0x02, 0x00, 0x00, 0x00,                         // section[1].id = 2 (BODY)
        0x48, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // section[1].offset = 72
        0x9B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // section[1].length = 155
        // BLOBS payload @ 52
        0x02, 0x00, 0x00, 0x00,                         // blob count = 2
        0x04, 0x00, 0x00, 0x00,                         // blob[0] len = 4
        0xDE, 0xAD, 0xBE, 0xEF,                         // blob[0] = cubin
        0x04, 0x00, 0x00, 0x00,                         // blob[1] len = 4
        0x0A, 0x00, 0x00, 0x00,                         // blob[1] = init (i32 10)
        // BODY payload @ 72
        0x01, 0x00, 0x00, 0x00,                         // module count = 1
        0x00, 0x00, 0x00, 0x00,                         // module[0] -> blob 0
        0x01, 0x00, 0x00, 0x00,                         // workspace count = 1
        0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // nbytes = 4
        0x01,                                           // has_init = 1
        0x01, 0x00, 0x00, 0x00,                         // init -> blob 1
        0x02, 0x00, 0x00, 0x00,                         // node count = 2
        // node[0] = Memset
        0x01,                                           // kind = 1 (Memset)
        0x00,                                           // target.tag = 0 (Output)
        0x00, 0x00, 0x00, 0x00,                         // target.index = 0
        0x00,                                           // value = 0
        0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // nbytes = 16
        0x00, 0x00, 0x00, 0x00,                         // deps count = 0
        // node[1] = KernelLaunch
        0x00,                                           // kind = 0 (KernelLaunch)
        0x00, 0x00, 0x00, 0x00,                         // module = 0
        0x04, 0x00, 0x00, 0x00,                         // entry len = 4
        0x66, 0x69, 0x6C, 0x6C,                         // entry = "fill"
        0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // grid
        0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // block
        0x00, 0x00, 0x00, 0x00,                         // shmem = 0
        0x03, 0x00, 0x00, 0x00,                         // args count = 3
        // arg[0] = Input(0) @ offset 0
        0x01, 0x00, 0x00, 0x00,                         //   segment count = 1
        0x01,                                           //   seg.tag = 1 (Pointer)
        0x00, 0x00, 0x00, 0x00, 0x00,                   //   bufid = Input(0)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //   offset = 0
        // arg[1] = Workspace(0) @ offset 0
        0x01, 0x00, 0x00, 0x00,                         //   segment count = 1
        0x01,                                           //   seg.tag = 1 (Pointer)
        0x02, 0x00, 0x00, 0x00, 0x00,                   //   bufid = Workspace(0)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //   offset = 0
        // arg[2] = Output(0) @ offset 0
        0x01, 0x00, 0x00, 0x00,                         //   segment count = 1
        0x01,                                           //   seg.tag = 1 (Pointer)
        0x01, 0x00, 0x00, 0x00, 0x00,                   //   bufid = Output(0)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, //   offset = 0
        0x01, 0x00, 0x00, 0x00,                         // deps count = 1
        0x00, 0x00, 0x00, 0x00,                         // deps[0] = 0
    ];

    #[test]
    fn golden_is_227_bytes() {
        assert_eq!(GOLDEN.len(), 227);
    }

    fn rich_plan() -> ExecutionPlan {
        ExecutionPlan {
            modules: vec![
                KernelModule {
                    cubin: Blob::from_vec(vec![0x01, 0x02, 0x03]),
                },
                KernelModule {
                    cubin: Blob::default(),
                },
            ],
            workspace: vec![
                WorkspaceBuf {
                    nbytes: 1 << 32,
                    init: None,
                },
                WorkspaceBuf {
                    nbytes: 64,
                    init: Some(Blob::from_vec(vec![9, 9, 9, 9, 9])),
                },
            ],
            nodes: vec![
                Node {
                    op: Op::Memset {
                        target: WritableBuf::Output(0),
                        value: 9,
                        nbytes: 1 << 33,
                    },
                    deps: vec![],
                },
                Node {
                    op: Op::KernelLaunch {
                        module: 1,
                        entry: "compute".into(),
                        grid: [7, 3, 1],
                        block: [128, 2, 1],
                        shmem: 4096,
                        args: vec![
                            Arg::input(2),
                            Arg::output(0),
                            Arg::workspace(1),
                            Arg::workspace(0),
                        ],
                    },
                    deps: vec![0],
                },
            ],
        }
    }

    #[test]
    fn round_trips_rich_plan() {
        let plan = rich_plan();
        assert_eq!(deserialize(&serialize(&plan).unwrap()), Ok(plan));
    }

    fn candidate_plan(entry: &str) -> ExecutionPlan {
        ExecutionPlan {
            modules: vec![KernelModule {
                cubin: Blob::from_vec(vec![0xAA, 0xBB]),
            }],
            workspace: vec![WorkspaceBuf {
                nbytes: 32,
                init: None,
            }],
            nodes: vec![Node {
                op: Op::KernelLaunch {
                    module: 0,
                    entry: entry.into(),
                    grid: [4, 1, 1],
                    block: [64, 1, 1],
                    shmem: 0,
                    args: vec![Arg::input(0), Arg::workspace(0), Arg::output(0)],
                },
                deps: vec![],
            }],
        }
    }

    fn choice_plan() -> ExecutionPlan {
        ExecutionPlan {
            modules: vec![],
            workspace: vec![WorkspaceBuf {
                nbytes: 64,
                init: None,
            }],
            nodes: vec![Node {
                op: Op::Choice {
                    candidates: vec![candidate_plan("fast"), candidate_plan("slow")],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Workspace(0)],
                },
                deps: vec![],
            }],
        }
    }

    #[test]
    fn round_trips_choice_plan() {
        let plan = choice_plan();
        assert_eq!(deserialize(&serialize(&plan).unwrap()), Ok(plan));
    }

    #[test]
    fn pool_shares_identical_payloads_across_candidates() {
        // Both candidates carry byte-identical cubins; the pool must hold the
        // bytes once even though the tree stores two modules.
        let bytes = serialize(&choice_plan()).unwrap();
        let one_candidate = serialize(&ExecutionPlan {
            nodes: vec![Node {
                op: Op::Choice {
                    candidates: vec![candidate_plan("fast")],
                    input_binding: vec![BufRef::Input(0)],
                    output_binding: vec![BufRef::Workspace(0)],
                },
                deps: vec![],
            }],
            ..choice_plan()
        })
        .unwrap();
        // The two-candidate wire adds only the second candidate's body (its
        // cubin is pooled), which is far smaller than the cubin would be.
        let body_only = bytes.len() - one_candidate.len();
        let candidate_body = serialize(&candidate_plan("slow")).unwrap().len();
        assert!(body_only < candidate_body);

        let decoded = deserialize(&bytes).unwrap();
        let Op::Choice { candidates, .. } = &decoded.nodes[0].op else {
            unreachable!()
        };
        assert_eq!(
            candidates[0].modules[0].cubin,
            candidates[1].modules[0].cubin
        );
    }

    #[test]
    fn validates_choice_plan() {
        assert_eq!(choice_plan().validate(1, 0), Ok(()));
    }

    #[test]
    fn rejects_empty_choice() {
        let mut plan = choice_plan();
        let Op::Choice { candidates, .. } = &mut plan.nodes[0].op else {
            unreachable!()
        };
        candidates.clear();
        assert_eq!(
            plan.validate(1, 0),
            Err(crate::execution_plan::PlanError::EmptyChoice { node: 0 })
        );
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = GOLDEN.to_vec();
        bytes[0] = b'X';
        assert_eq!(
            deserialize(&bytes),
            Err(WireError::BadMagic {
                found: [b'X', 0x43, 0x50, 0x45]
            })
        );
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = GOLDEN.to_vec();
        bytes[4] = 99;
        assert_eq!(
            deserialize(&bytes),
            Err(WireError::UnsupportedVersion { found: 99 })
        );
    }

    #[test]
    fn rejects_truncated() {
        assert_eq!(deserialize(&GOLDEN[..100]), Err(WireError::Truncated));
        assert_eq!(deserialize(&[1, 2]), Err(WireError::Truncated));
    }

    #[test]
    fn rejects_bad_node_kind() {
        let mut bytes = GOLDEN.to_vec();
        // BODY @ 72: modules (8) + workspace (4 + 13) + node count (4).
        let node_kind = 72 + 8 + 17 + 4;
        assert_eq!(bytes[node_kind], 1); // Memset, per the fixture
        bytes[node_kind] = 7;
        assert_eq!(
            deserialize(&bytes),
            Err(WireError::BadTag {
                what: "node kind",
                tag: 7
            })
        );
    }

    #[test]
    fn rejects_blob_index_out_of_range() {
        let mut bytes = GOLDEN.to_vec();
        let module_pool_index = 72 + 4;
        bytes[module_pool_index] = 9;
        assert_eq!(
            deserialize(&bytes),
            Err(WireError::BlobIndexOutOfRange { index: 9 })
        );
    }

    #[test]
    fn checked_u32_rejects_oversize() {
        let value = u32::MAX as usize + 1;
        assert_eq!(
            checked_u32(value, "test field"),
            Err(WireError::Oversize {
                what: "test field",
                value: value as u64,
            })
        );
    }

    #[test]
    fn checked_u32_accepts_u32_max() {
        assert_eq!(checked_u32(u32::MAX as usize, "test field"), Ok(u32::MAX));
    }

    #[test]
    fn deserialize_shared_views_the_source() {
        let plan = choice_plan();
        let bytes = serialize(&plan).unwrap();
        let decoded = deserialize_shared(Blob::from_vec(bytes)).unwrap();
        assert_eq!(decoded, plan);
        let mut detached = decoded.clone();
        detached.detach_blobs();
        assert_eq!(detached, plan);
    }
}
