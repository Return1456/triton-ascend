// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --split-input-file %s | FileCheck %s

// bool stores are represented as ptr<i1> -> ptr<i8> because i1 occupies one
// byte in the Ascend ABI. Push the cast below the broadcast so PtrAnalysis
// sees the ordinary splat/addptr/broadcast chain.
module {
  tt.func public @push_bool_store_bitcast(%out: !tt.ptr<i1>, %offset: tensor<16x1xi32>, %value: tensor<16x1xi8>, %mask: tensor<16x1xi1>) {
    %ptr = tt.splat %out : !tt.ptr<i1> -> tensor<16x1x!tt.ptr<i1>>
    %add = tt.addptr %ptr, %offset : tensor<16x1x!tt.ptr<i1>>, tensor<16x1xi32>
    %cast = tt.bitcast %add : tensor<16x1x!tt.ptr<i1>> -> tensor<16x1x!tt.ptr<i8>>
    tt.store %cast, %value, %mask : tensor<16x1x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @push_bool_store_bitcast
// CHECK: %[[BASE:.*]] = tt.bitcast %{{.*}} : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK: %[[SPLAT:.*]] = tt.splat %[[BASE]] : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
// CHECK: %[[ADD:.*]] = tt.addptr %[[SPLAT]], %{{.*}} : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
// CHECK: tt.store %[[ADD]], %{{.*}}, %{{.*}} : tensor<16x1x!tt.ptr<i8>>
// CHECK-NOT: tt.bitcast %[[ADD]]

// -----

// A shared bitcast source must stay alive when it has other users.
module {
  tt.func public @shared_source(%out: !tt.ptr<i1>, %offset: tensor<8xi32>) -> (tensor<8x!tt.ptr<i1>>, tensor<8x!tt.ptr<i8>>) {
    %ptr = tt.splat %out : !tt.ptr<i1> -> tensor<8x!tt.ptr<i1>>
    %add = tt.addptr %ptr, %offset : tensor<8x!tt.ptr<i1>>, tensor<8xi32>
    %cast = tt.bitcast %add : tensor<8x!tt.ptr<i1>> -> tensor<8x!tt.ptr<i8>>
    tt.return %add, %cast : tensor<8x!tt.ptr<i1>>, tensor<8x!tt.ptr<i8>>
  }
}

// CHECK-LABEL: tt.func public @shared_source
// CHECK: %[[ADD_I1:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<i1>>, tensor<8xi32>
// CHECK: %[[BASE_I8:.*]] = tt.bitcast %{{.*}} : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK: %[[SPLAT_I8:.*]] = tt.splat %[[BASE_I8]] : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
// CHECK: %[[ADD_I8:.*]] = tt.addptr %[[SPLAT_I8]], %{{.*}} : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
// CHECK: tt.return %[[ADD_I1]], %[[ADD_I8]]

// -----

// A width-changing pointer cast is not rewritten.
module {
  tt.func public @reject_width_change(%src: !tt.ptr<i8>, %offset: tensor<8xi32>) -> tensor<8x!tt.ptr<f32>> {
    %ptr = tt.splat %src : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %add = tt.addptr %ptr, %offset : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %cast = tt.bitcast %add : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<f32>>
    tt.return %cast : tensor<8x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: tt.func public @reject_width_change
// CHECK: %[[ADD:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
// CHECK: %[[CAST:.*]] = tt.bitcast %[[ADD]] : tensor<8x!tt.ptr<i8>> -> tensor<8x!tt.ptr<f32>>
// CHECK: tt.return %[[CAST]]
