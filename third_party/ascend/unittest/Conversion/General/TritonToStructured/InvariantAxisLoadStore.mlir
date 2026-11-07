// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --split-input-file %s | FileCheck %s

// Regression tests for the generalized zero-stride (broadcast) load/store
// rewrite. The old converters only matched a `tt.broadcast` that directly fed
// `tt.load`/`tt.store`; these cases hide the broadcast behind a pointer
// bitcast, a second addptr, a loop-carried pointer or an `scf.if` result.

// -----

// The A5 failure shape: bool pointer broadcast -> ptr<i1> to ptr<i8> bitcast ->
// load. The load must be narrowed to the representative [16x1] slice and
// broadcast back, keeping `was_bool_to_int8` on the small load.
module {
  tt.func public @bool_broadcast_bitcast_load(%src: !tt.ptr<i1>, %dst: !tt.ptr<i8>, %M: i32) attributes {noinline = false} {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %sp = tt.splat %src : !tt.ptr<i1> -> tensor<16x1x!tt.ptr<i1>>
    %ap = tt.addptr %sp, %r1 : tensor<16x1x!tt.ptr<i1>>, tensor<16x1xi32>
    %bp = tt.broadcast %ap : tensor<16x1x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i1>>
    %bc = tt.bitcast %bp : tensor<16x16x!tt.ptr<i1>> -> tensor<16x16x!tt.ptr<i8>>
    %m1 = tt.splat %M : i32 -> tensor<16x1xi32>
    %cmp = arith.cmpi slt, %r1, %m1 : tensor<16x1xi32>
    %m = tt.broadcast %cmp : tensor<16x1xi1> -> tensor<16x16xi1>
    %zero = arith.constant dense<0> : tensor<16x16xi8>
    %v = tt.load %bc, %m, %zero {was_bool_to_int8 = true} : tensor<16x16x!tt.ptr<i8>>
    %ds = tt.splat %dst : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %dp1 = tt.addptr %ds, %r1 : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %dp = tt.broadcast %dp1 : tensor<16x1x!tt.ptr<i8>> -> tensor<16x16x!tt.ptr<i8>>
    tt.store %dp, %v : tensor<16x16x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @bool_broadcast_bitcast_load
// CHECK: tt.load %{{.*}} {was_bool_to_int8 = true} : tensor<16x1x!tt.ptr<i8>>
// CHECK: tt.broadcast %{{.*}} : tensor<16x1xi8> -> tensor<16x16xi8>
// CHECK: tensor.extract_slice %{{.*}}[0, 0] [16, 1] [1, 1] : tensor<16x16xi8> to tensor<16x1xi8>
// CHECK: tt.store %{{.*}} : tensor<16x1x!tt.ptr<i8>>

// -----

// Multi-level addptr: the broadcast pointer is followed by a second addptr
// whose offset is broadcast along axis 1. Axis 0 is the only varying axis, so
// axis 1 is compressed and the small load is broadcast back.
module {
  tt.func public @multi_level_addptr(%X: !tt.ptr<f32>, %Out: !tt.ptr<f32>, %s: i32) attributes {noinline = false} {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %sp = tt.splat %s : i32 -> tensor<16x1xi32>
    %rowoff = arith.muli %r1, %sp : tensor<16x1xi32>
    %rowb = tt.broadcast %rowoff : tensor<16x1xi32> -> tensor<16x16xi32>
    %ps = tt.splat %X : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %p1 = tt.addptr %ps, %r1 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %pb = tt.broadcast %p1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %p2 = tt.addptr %pb, %rowb : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %v = tt.load %p2 : tensor<16x16x!tt.ptr<f32>>
    %os = tt.splat %Out : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %op = tt.addptr %os, %rowb : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %op, %v : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @multi_level_addptr
// CHECK: tensor.extract_slice %{{.*}}[0, 0] [16, 1] [1, 1] : tensor<16x16x!tt.ptr<f32>> to tensor<16x1x!tt.ptr<f32>>
// CHECK: tt.load %{{.*}} : tensor<16x1x!tt.ptr<f32>>
// CHECK: tt.broadcast %{{.*}} : tensor<16x1xf32> -> tensor<16x16xf32>
// CHECK: tensor.extract_slice %{{.*}}[0, 0] [16, 1] [1, 1] : tensor<16x16xf32> to tensor<16x1xf32>
// CHECK: tt.store %{{.*}} : tensor<16x1x!tt.ptr<f32>>

// -----

// Loop-carried pointer: only the initial value and the backedge are needed to
// prove that axis 0 never varies, so the load inside the loop is compressed.
module {
  tt.func public @iter_arg_ptr(%arg0: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32, %s: i32) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %e = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %o = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %sp = tt.splat %s : i32 -> tensor<16x1xi32>
    %rowoff = arith.muli %e, %sp : tensor<16x1xi32>
    %p0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %a0 = tt.addptr %p0, %rowoff : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %bc = tt.broadcast %a0 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %acc0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %res:2 = scf.for %i = %c0 to %n step %c1 iter_args(%p = %bc, %acc = %acc0) -> (tensor<16x16x!tt.ptr<f32>>, tensor<16x16xf32>) : i32 {
      %v = tt.load %p : tensor<16x16x!tt.ptr<f32>>
      %a = arith.addf %acc, %v : tensor<16x16xf32>
      %idx = tt.splat %i : i32 -> tensor<16x1xi32>
      %off = arith.muli %idx, %sp : tensor<16x1xi32>
      %offb = tt.broadcast %off : tensor<16x1xi32> -> tensor<16x16xi32>
      %p2 = tt.addptr %p, %offb : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      scf.yield %p2, %a : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xf32>
    }
    %c16 = arith.constant dense<16> : tensor<16x1xi32>
    %row = arith.muli %e, %c16 : tensor<16x1xi32>
    %rowb = tt.broadcast %row : tensor<16x1xi32> -> tensor<16x16xi32>
    %ob = tt.broadcast %o : tensor<1x16xi32> -> tensor<16x16xi32>
    %off2 = arith.addi %rowb, %ob : tensor<16x16xi32>
    %os = tt.splat %out : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %op = tt.addptr %os, %off2 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %op, %res#1 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @iter_arg_ptr
// CHECK: scf.for
// CHECK: tensor.extract_slice %{{.*}}[0, 0] [16, 1] [1, 1] : tensor<16x16x!tt.ptr<f32>> to tensor<16x1x!tt.ptr<f32>>
// CHECK: tt.load %{{.*}} : tensor<16x1x!tt.ptr<f32>>
// CHECK: tt.broadcast %{{.*}} : tensor<16x1xf32> -> tensor<16x16xf32>

// -----

// scf.if result: both branches yield broadcast pointers, so the intersection
// still proves axis 0 invariant and the load after the if is compressed.
module {
  tt.func public @if_result_ptr(%X: !tt.ptr<f32>, %Y: !tt.ptr<f32>, %c: i1, %out: !tt.ptr<f32>) attributes {noinline = false} {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %r0 = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %c16 = arith.constant dense<16> : tensor<16x1xi32>
    %row = arith.muli %r1, %c16 : tensor<16x1xi32>
    %rowb = tt.broadcast %row : tensor<16x1xi32> -> tensor<16x16xi32>
    %ob = tt.broadcast %r0 : tensor<1x16xi32> -> tensor<16x16xi32>
    %off = arith.addi %rowb, %ob : tensor<16x16xi32>
    %p = scf.if %c -> (tensor<16x16x!tt.ptr<f32>>) {
      %sp = tt.splat %X : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
      %a = tt.addptr %sp, %r1 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
      %b = tt.broadcast %a : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
      scf.yield %b : tensor<16x16x!tt.ptr<f32>>
    } else {
      %sp2 = tt.splat %Y : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
      %a2 = tt.addptr %sp2, %r1 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
      %b2 = tt.broadcast %a2 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
      scf.yield %b2 : tensor<16x16x!tt.ptr<f32>>
    }
    %v = tt.load %p : tensor<16x16x!tt.ptr<f32>>
    %os = tt.splat %out : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %op = tt.addptr %os, %off : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %op, %v : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @if_result_ptr
// CHECK: scf.if
// CHECK: tensor.extract_slice %{{.*}}[0, 0] [16, 1] [1, 1] : tensor<16x16x!tt.ptr<f32>> to tensor<16x1x!tt.ptr<f32>>
// CHECK: tt.load %{{.*}} : tensor<16x1x!tt.ptr<f32>>
// CHECK: tt.broadcast %{{.*}} : tensor<16x1xf32> -> tensor<16x16xf32>

// -----

// Negative: the pointer genuinely varies on both axes, so the load and the
// store must keep their original full shape.
module {
  tt.func public @varying_ptr_untouched(%X: !tt.ptr<f32>, %Out: !tt.ptr<f32>) attributes {noinline = false} {
    %r = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %r0 = tt.expand_dims %r {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %s1 = tt.splat %X : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %p1 = tt.addptr %s1, %r1 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %pb = tt.broadcast %p1 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %ob = tt.broadcast %r0 : tensor<1x16xi32> -> tensor<16x16xi32>
    %p2 = tt.addptr %pb, %ob : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %v = tt.load %p2 : tensor<16x16x!tt.ptr<f32>>
    tt.store %p2, %v : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @varying_ptr_untouched
// CHECK-NOT: tensor.extract_slice %{{.*}} : tensor<16x16x!tt.ptr<f32>>
// CHECK: tt.load %{{.*}} : tensor<16x16x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}} : tensor<16x16x!tt.ptr<f32>>