/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef TRITON_TO_STRUCTURED_INVARIANTAXISANALYSIS_H
#define TRITON_TO_STRUCTURED_INVARIANTAXISANALYSIS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace TritonToStructured {
using namespace mlir;

/// Proves which tensor dimensions do not contribute to the element values of a
/// pointer (or integer offset) tensor. A `true` entry means every element along
/// that dimension holds the same value, i.e. the corresponding address stride
/// is statically zero. Dimensions of size one are trivially invariant.
///
/// Straight-line expressions are handled structurally: splat, broadcast,
/// expand_dims, elementwise arithmetic, pointer addptr, pointer bitcast and
/// extract_slice all propagate facts. Control flow extends the same facts
/// through `scf.for` iter_args (a fixed point over the initial value and the
/// backedge) and `scf.if` results (the intersection of both branches).
///
/// The analysis is deliberately conservative: anything it cannot prove is
/// reported as varying, so callers may compress only proven-invariant axes and
/// never risk merging addresses that actually differ. It creates no IR, so it
/// is safe to run inside a rewrite pattern.
class InvariantAxisAnalysis {
public:
  using AxisMask = SmallVector<bool>;

  struct AxisFacts {
    SmallVector<bool> invariant;
  };

  InvariantAxisAnalysis() = default;

  /// Returns one flag per tensor dimension. Scalars produce an empty mask.
  FailureOr<AxisMask> analyze(Value value);

private:
  FailureOr<AxisFacts> analyzeFacts(Value value);
  FailureOr<AxisFacts> analyzeFactsOrVarying(Value value,
                                             ArrayRef<int64_t> shape);
  FailureOr<AxisFacts> analyzeOperation(Value value);
  FailureOr<AxisFacts> analyzeForIterArg(scf::ForOp forOp, unsigned index);
  FailureOr<AxisFacts> analyzeForResult(scf::ForOp forOp, unsigned resultIndex);
  FailureOr<AxisFacts> analyzeIfResult(scf::IfOp ifOp, unsigned resultIndex);
  FailureOr<AxisFacts> analyzeExtractSlice(tensor::ExtractSliceOp slice,
                                           ArrayRef<int64_t> shape);

  DenseMap<Value, AxisFacts> cache;
  DenseMap<Value, AxisFacts> bindings;
  SmallPtrSet<Value, 16> active;
};

} // namespace TritonToStructured

#endif // TRITON_TO_STRUCTURED_INVARIANTAXISANALYSIS_H