// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::cuda_driver::ffi::{
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE, CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_HANDLE_TYPE_NONE,
    CU_MEM_LOCATION_TYPE_DEVICE, CUmemAccessDesc, CUmemAllocationProp, CUmemLocation,
};
use crate::cuda_driver::{
    CUcontext, CUdevice, CUdeviceptr, CUstream, Context, CudaDriver, DeviceBuffer, check,
};
use crate::execution_plan::{
    Arg, BufRef, ExecutionPlan, FlatPlan, KernelModule, Node, Op, WritableBuf, deserialize,
    serialize,
};
use crate::plan_executor::{Bindings, ResidentPlan, execute, tune};

const FILL_PTX: &str = r#".version 7.0
.target sm_52
.address_size 64
.visible .entry fill(
    .param .u64 fill_param_0
)
{
    .reg .pred %p1;
    .reg .f32 %f1;
    .reg .b32 %r<6>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [fill_param_0];
    mov.f32 %f1, 0f40E00000;
    mov.u32 %r1, 256;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r3, %r2, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra $L_DONE;
    cvta.to.global.u64 %rd2, %rd1;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f1;
$L_DONE:
    ret;
}
"#;

const N: usize = 256;
const NBYTES: usize = N * 4;
const FILL_VALUE: f32 = 7.0;

fn fill_ptx_cubin() -> Vec<u8> {
    let mut bytes = FILL_PTX.as_bytes().to_vec();
    bytes.push(0);
    bytes
}

fn fill_plan(grid: [u32; 3]) -> ExecutionPlan {
    ExecutionPlan {
        modules: vec![KernelModule {
            cubin: fill_ptx_cubin().into(),
        }],
        workspace: vec![],
        nodes: vec![
            Node {
                op: Op::Memset {
                    target: WritableBuf::Output(0),
                    value: 0,
                    nbytes: NBYTES,
                },
                deps: vec![],
            },
            Node {
                op: Op::KernelLaunch {
                    module: 0,
                    entry: "fill".into(),
                    grid,
                    block: [256, 1, 1],
                    shmem: 0,
                    args: vec![Arg::output(0)],
                },
                deps: vec![0],
            },
        ],
    }
}

fn choice_fill_plan() -> ExecutionPlan {
    let candidate = |grid: [u32; 3]| ExecutionPlan {
        modules: vec![KernelModule {
            cubin: fill_ptx_cubin().into(),
        }],
        workspace: vec![],
        nodes: vec![Node {
            op: Op::KernelLaunch {
                module: 0,
                entry: "fill".into(),
                grid,
                block: [256, 1, 1],
                shmem: 0,
                args: vec![Arg::output(0)],
            },
            deps: vec![],
        }],
    };
    ExecutionPlan {
        modules: vec![],
        workspace: vec![],
        nodes: vec![
            Node {
                op: Op::Memset {
                    target: WritableBuf::Output(0),
                    value: 0,
                    nbytes: NBYTES,
                },
                deps: vec![],
            },
            Node {
                op: Op::Choice {
                    candidates: vec![candidate([1, 1, 1]), candidate([8, 1, 1])],
                    input_binding: vec![],
                    output_binding: vec![BufRef::Output(0)],
                },
                deps: vec![0],
            },
        ],
    }
}

unsafe fn init_context(drv: &CudaDriver) {
    let r = unsafe { (drv.cuInit)(0) };
    check(r, "cuInit").unwrap();
    let mut dev: CUdevice = 0;
    let r = unsafe { (drv.cuDeviceGet)(&mut dev, 0) };
    check(r, "cuDeviceGet").unwrap();
    let mut ctx: CUcontext = std::ptr::null_mut();
    let r = unsafe { (drv.cuCtxCreate_v2)(&mut ctx, 0, dev) };
    check(r, "cuCtxCreate_v2").unwrap();
    let r = unsafe { (drv.cuCtxSetCurrent)(ctx) };
    check(r, "cuCtxSetCurrent").unwrap();
}

unsafe fn memset(drv: &CudaDriver, dev: CUdeviceptr, value: u8, stream: CUstream) {
    let r = unsafe { (drv.cuMemsetD8Async)(dev, value, NBYTES, stream) };
    check(r, "cuMemsetD8Async (reset)").unwrap();
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (reset)").unwrap();
}

fn assert_all_filled(label: &str, out: &[f32]) {
    assert!(
        out.iter().all(|&x| x == FILL_VALUE),
        "{label}: output buffer not uniformly {FILL_VALUE}: {out:?}"
    );
    println!("{label}: out[0]={}, all {N} == {FILL_VALUE}", out[0]);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn fill_plan_runs() {
    let drv = CudaDriver::get();

    let r = unsafe { (drv.cuInit)(0) };
    check(r, "cuInit").unwrap();

    let mut dev: CUdevice = 0;
    let r = unsafe { (drv.cuDeviceGet)(&mut dev, 0) };
    check(r, "cuDeviceGet").unwrap();

    let mut ctx: CUcontext = std::ptr::null_mut();
    let r = unsafe { (drv.cuCtxCreate_v2)(&mut ctx, 0, dev) };
    check(r, "cuCtxCreate_v2").unwrap();
    let r = unsafe { (drv.cuCtxSetCurrent)(ctx) };
    check(r, "cuCtxSetCurrent").unwrap();

    let out = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let ws = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();

    let stream: CUstream = std::ptr::null_mut();

    let bindings = Bindings {
        inputs: &[],
        outputs: &[out.ptr()],
        workspace: ws.ptr(),
    };

    let plan = fill_plan([1, 1, 1]);
    assert_eq!(plan.validate(0, 1), Ok(()));
    let mut loaded = unsafe {
        ResidentPlan::new(
            ctx,
            FlatPlan::assume_flat(deserialize(&serialize(&plan).unwrap()).unwrap()),
        )
    }
    .expect("ResidentPlan::new");

    unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute");
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (exec)").unwrap();
    assert_all_filled("execute/Static", &out.to_host().unwrap());

    unsafe { memset(drv, out.ptr(), 0, stream) };
    unsafe { memset(drv, ws.ptr(), 0, stream) };
    let wide_plan = fill_plan([8, 1, 1]);
    assert_eq!(wide_plan.validate(0, 1), Ok(()));
    let mut wide_loaded = unsafe {
        ResidentPlan::new(
            ctx,
            FlatPlan::assume_flat(deserialize(&serialize(&wide_plan).unwrap()).unwrap()),
        )
    }
    .expect("ResidentPlan::new (wide grid)");
    unsafe { execute(&mut wide_loaded, &bindings, stream) }.expect("execute (wide grid)");
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (wide grid)").unwrap();
    assert_all_filled("execute/wide-grid", &out.to_host().unwrap());
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_exec_is_repointed_across_buffers() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };

    let stream: CUstream = std::ptr::null_mut();
    let ctx = Context::current().raw();
    let ws = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();

    let plan = fill_plan([1, 1, 1]);
    assert_eq!(plan.validate(0, 1), Ok(()));
    let mut loaded = unsafe {
        ResidentPlan::new(
            ctx,
            FlatPlan::assume_flat(deserialize(&serialize(&plan).unwrap()).unwrap()),
        )
    }
    .expect("ResidentPlan::new");

    let outs: Vec<DeviceBuffer<f32>> = (0..3)
        .map(|_| DeviceBuffer::<f32>::alloc(ctx, N).unwrap())
        .collect();
    for out in &outs {
        unsafe { memset(drv, out.ptr(), 0, stream) };
        let bindings = Bindings {
            inputs: &[],
            outputs: &[out.ptr()],
            workspace: ws.ptr(),
        };
        unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (re-point)");
        let r = unsafe { (drv.cuStreamSynchronize)(stream) };
        check(r, "cuStreamSynchronize (re-point)").unwrap();
    }

    for (i, out) in outs.iter().enumerate() {
        let out = out.to_host().unwrap();
        assert!(
            out.iter().all(|&x| x == FILL_VALUE),
            "buffer {i} not uniformly {FILL_VALUE}: {out:?}"
        );
        println!(
            "re-point buffer {i}: out[0]={}, all {N} == {FILL_VALUE}",
            out[0]
        );
    }
}

/// Reserve, back, map and enable a VMM region (`cuMemCreate`/`cuMemMap`) —
/// the mapping kind XLA uses for some of its arenas, distinct from
/// `cuMemAlloc`. Leaked at test end.
unsafe fn vmm_alloc(drv: &CudaDriver, min_size: usize) -> CUdeviceptr {
    let prop = CUmemAllocationProp {
        type_: CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: CU_MEM_HANDLE_TYPE_NONE,
        location: CUmemLocation {
            type_: CU_MEM_LOCATION_TYPE_DEVICE,
            id: 0,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: unsafe { std::mem::zeroed() },
    };
    let mut gran = 0usize;
    let r = unsafe { (drv.cuMemGetAllocationGranularity)(&mut gran, &prop, 0) };
    check(r, "cuMemGetAllocationGranularity").unwrap();
    let size = min_size.next_multiple_of(gran);
    let mut base: CUdeviceptr = 0;
    let r = unsafe { (drv.cuMemAddressReserve)(&mut base, size, 0, 0, 0) };
    check(r, "cuMemAddressReserve").unwrap();
    let mut handle = 0u64;
    let r = unsafe { (drv.cuMemCreate)(&mut handle, size, &prop, 0) };
    check(r, "cuMemCreate").unwrap();
    let r = unsafe { (drv.cuMemMap)(base, size, 0, handle, 0) };
    check(r, "cuMemMap").unwrap();
    let desc = CUmemAccessDesc {
        location: CUmemLocation {
            type_: CU_MEM_LOCATION_TYPE_DEVICE,
            id: 0,
        },
        flags: CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
    };
    let r = unsafe { (drv.cuMemSetAccess)(base, size, &desc, 1) };
    check(r, "cuMemSetAccess").unwrap();
    base
}

unsafe fn to_host_f32(drv: &CudaDriver, src: CUdeviceptr, stream: CUstream) -> Vec<f32> {
    let mut host = vec![0f32; N];
    let r = unsafe { (drv.cuMemcpyDtoHAsync_v2)(host.as_mut_ptr() as *mut _, src, NBYTES, stream) };
    check(r, "cuMemcpyDtoHAsync_v2").unwrap();
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (to_host)").unwrap();
    host
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_exec_repoints_to_vmm_mapped_buffers() {
    // The resident graph is built on cuMemAlloc'd placeholders; XLA can then
    // bind buffers from a cuMemMap'd arena.
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };

    let stream: CUstream = std::ptr::null_mut();
    let ctx = Context::current().raw();

    let region = unsafe { vmm_alloc(drv, 2 * NBYTES) };
    let out = region;
    let ws = region + NBYTES as CUdeviceptr;

    let plan = fill_plan([1, 1, 1]);
    assert_eq!(plan.validate(0, 1), Ok(()));
    let mut loaded = unsafe {
        ResidentPlan::new(
            ctx,
            FlatPlan::assume_flat(deserialize(&serialize(&plan).unwrap()).unwrap()),
        )
    }
    .expect("ResidentPlan::new (vmm)");

    unsafe { memset(drv, out, 0, stream) };
    unsafe { memset(drv, ws, 0, stream) };
    let bindings = Bindings {
        inputs: &[],
        outputs: &[out],
        workspace: ws,
    };
    unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (vmm)");
    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (vmm)").unwrap();

    let out = unsafe { to_host_f32(drv, out, stream) };
    assert_all_filled("execute/vmm", &out);
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn graph_ring_pipelines_without_intercall_sync() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };

    let stream: CUstream = std::ptr::null_mut();
    let ctx = Context::current().raw();
    let ws = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();

    let plan = fill_plan([1, 1, 1]);
    assert_eq!(plan.validate(0, 1), Ok(()));
    let mut loaded = unsafe {
        ResidentPlan::new(
            ctx,
            FlatPlan::assume_flat(deserialize(&serialize(&plan).unwrap()).unwrap()),
        )
    }
    .expect("ResidentPlan::new");

    const CALLS: usize = 8;
    let outs: Vec<DeviceBuffer<f32>> = (0..CALLS)
        .map(|_| DeviceBuffer::<f32>::alloc(ctx, N).unwrap())
        .collect();
    for out in &outs {
        unsafe { memset(drv, out.ptr(), 0, stream) };
    }

    for out in &outs {
        let bindings = Bindings {
            inputs: &[],
            outputs: &[out.ptr()],
            workspace: ws.ptr(),
        };
        unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (pipelined)");
    }

    let r = unsafe { (drv.cuStreamSynchronize)(stream) };
    check(r, "cuStreamSynchronize (pipelined)").unwrap();
    for (i, out) in outs.iter().enumerate() {
        let out = out.to_host().unwrap();
        assert!(
            out.iter().all(|&x| x == FILL_VALUE),
            "pipelined buffer {i} not uniformly {FILL_VALUE}: {out:?}"
        );
        println!(
            "pipelined buffer {i}: out[0]={}, all {N} == {FILL_VALUE}",
            out[0]
        );
    }
}

const SPIN_PTX: &str = r#".version 7.0
.target sm_52
.address_size 64
.visible .entry spin(
    .param .u64 spin_param_0
)
{
    .reg .pred %p1;
    .reg .b64 %rd<7>;
    ld.param.u64 %rd1, [spin_param_0];
    mov.u64 %rd2, 3000000;
    cvta.to.global.u64 %rd3, %rd1;
    mov.u64 %rd4, %clock64;
$L_SPIN:
    mov.u64 %rd5, %clock64;
    sub.s64 %rd6, %rd5, %rd4;
    setp.lt.s64 %p1, %rd6, %rd2;
    @%p1 bra $L_SPIN;
    st.global.u64 [%rd3], %rd6;
    ret;
}
"#;

fn spin_plan(kernels: usize) -> ExecutionPlan {
    let mut bytes = SPIN_PTX.as_bytes().to_vec();
    bytes.push(0);
    let node = |output: usize| Node {
        op: Op::KernelLaunch {
            module: 0,
            entry: "spin".into(),
            grid: [16, 1, 1],
            block: [32, 1, 1],
            shmem: 0,
            args: vec![Arg::output(output)],
        },
        deps: vec![],
    };
    ExecutionPlan {
        modules: vec![KernelModule {
            cubin: bytes.into(),
        }],
        workspace: vec![],
        nodes: (0..kernels).map(node).collect(),
    }
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn independent_nodes_overlap_in_the_graph() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let stream: CUstream = std::ptr::null_mut();
    let ctx = Context::current().raw();

    let outs: Vec<DeviceBuffer<u64>> = (0..2)
        .map(|_| DeviceBuffer::<u64>::alloc(ctx, 1).unwrap())
        .collect();
    let out_ptrs: Vec<CUdeviceptr> = outs.iter().map(|out| out.ptr()).collect();

    let time = |kernels: usize| {
        let plan = spin_plan(kernels);
        assert_eq!(plan.validate(0, kernels), Ok(()));
        let mut loaded = unsafe { ResidentPlan::new(ctx, FlatPlan::assume_flat(plan)) }.unwrap();
        let bindings = Bindings {
            inputs: &[],
            outputs: &out_ptrs[..kernels],
            workspace: 0,
        };
        let mut best = f64::INFINITY;
        for _ in 0..4 {
            let start = std::time::Instant::now();
            unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (spin)");
            let r = unsafe { (drv.cuStreamSynchronize)(stream) };
            check(r, "cuStreamSynchronize (spin)").unwrap();
            best = best.min(start.elapsed().as_secs_f64());
        }
        best
    };

    let single = time(1);
    let pair = time(2);
    println!(
        "spin single {:.3}ms, independent pair {:.3}ms",
        single * 1e3,
        pair * 1e3
    );
    assert!(
        pair < single * 1.5,
        "independent graph nodes serialized: single {:.3}ms, pair {:.3}ms",
        single * 1e3,
        pair * 1e3
    );
}

#[test]
#[ignore = "requires an NVIDIA GPU"]
fn choice_tunes_and_runs() {
    let drv = CudaDriver::get();
    unsafe { init_context(drv) };
    let stream: CUstream = std::ptr::null_mut();

    let plan = choice_fill_plan();
    assert_eq!(plan.validate(0, 1), Ok(()));

    let tune_out = DeviceBuffer::<f32>::alloc(Context::current().raw(), N).unwrap();
    let tune_bindings = Bindings {
        inputs: &[],
        outputs: &[tune_out.ptr()],
        workspace: 0,
    };
    let flat = unsafe { tune(&plan, &tune_bindings, stream) }.expect("tune");
    assert!(
        flat.nodes
            .iter()
            .all(|n| !matches!(n.op, Op::Choice { .. })),
        "tuned plan is Choice-free"
    );
    let ctx = Context::current().raw();
    let mut loaded = unsafe { ResidentPlan::new(ctx, flat) }.expect("new (flat)");

    let mut run = |out: &DeviceBuffer<f32>| {
        let bindings = Bindings {
            inputs: &[],
            outputs: &[out.ptr()],
            workspace: 0,
        };
        // Dirty the real output so only a real run can leave the fill value.
        unsafe { memset(drv, out.ptr(), 0xff, stream) };
        unsafe { execute(&mut loaded, &bindings, stream) }.expect("execute (choice)");
        let r = unsafe { (drv.cuStreamSynchronize)(stream) };
        check(r, "cuStreamSynchronize (choice)").unwrap();
        out.to_host().unwrap()
    };

    let a = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    let b = DeviceBuffer::<f32>::alloc(ctx, N).unwrap();
    for (label, out_buf) in [("first", &a), ("re-point", &b)] {
        let out = run(out_buf);
        assert!(
            out.iter().all(|&x| x == FILL_VALUE),
            "choice/{label}: output not uniformly {FILL_VALUE}: {out:?}"
        );
        println!("choice/{label}: out[0]={}, all {N} == {FILL_VALUE}", out[0]);
    }
}
