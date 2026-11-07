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

#include "TritonToStructured/InvariantAxisAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace mlir::triton;

namespace TritonToStructured {

namespace {

InvariantAxisAnalysis::AxisFacts makeFacts(unsigned rank, bool invariant) {
  InvariantAxisAnalysis::AxisFacts facts;
  facts.invariant.assign(rank, invariant);
  return facts;
}

FailureOr<SmallVector<int64_t>> getStaticShape(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape())
    return failure();
  return SmallVector<int64_t>(type.getShape());
}

bool isPointerTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && isa<triton::PointerType>(type.getElementType());
}

bool isIntegerTensor(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getElementType().isIntOrIndex();
}

// Elementwise integer/pointer arithmetic cannot introduce variation along a
// dimension unless one of its inputs already varies there.
bool isElementwiseArithmetic(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
             arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::AndIOp,
             arith::OrIOp, arith::XOrIOp, arith::MaxSIOp, arith::MaxUIOp,
             arith::MinSIOp, arith::MinUIOp, arith::ShLIOp, arith::ShRUIOp,
             arith::ShRSIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
             arith::IndexCastOp, arith::IndexCastUIOp, arith::CmpIOp,
             arith::SelectOp, triton::AddPtrOp>(op);
}

FailureOr<InvariantAxisAnalysis::AxisFacts>
joinFacts(ArrayRef<InvariantAxisAnalysis::AxisFacts> inputs, unsigned rank) {
  InvariantAxisAnalysis::AxisFacts joined = makeFacts(rank, true);
  bool sawTensor = false;
  for (const auto &input : inputs) {
    if (input.invariant.empty())
      continue;
    if (input.invariant.size() != rank)
      return failure();
    sawTensor = true;
    for (auto [slot, bit] : llvm::zip(joined.invariant, input.invariant))
      slot = slot && bit;
  }
  if (!sawTensor)
    return makeFacts(rank, true);
  return joined;
}

} // namespace

FailureOr<InvariantAxisAnalysis::AxisMask>
InvariantAxisAnalysis::analyze(Value value) {
  if (!value)
    return failure();
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return AxisMask{};
  auto facts = analyzeFacts(value);
  if (failed(facts))
    return failure();
  if (facts->invariant.size() != static_cast<size_t>(type.getRank()))
    return failure();
  return AxisMask(facts->invariant.begin(), facts->invariant.end());
}

FailureOr<InvariantAxisAnalysis::AxisFacts>
InvariantAxisAnalysis::analyzeFacts(Value value) {
  if (!value)
    return failure();
  if (auto found = bindings.find(value); found != bindings.end())
    return found->second;
  if (auto found = cache.find(value); found != cache.end())
    return found->second;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return failure();
    auto iterArgs = forOp.getRegionIterArgs();
    auto iter = llvm::find(iterArgs, blockArg);
    if (iter == iterArgs.end())
      return failure();
    return analyzeForIterArg(forOp, std::distance(iterArgs.begin(), iter));
  }

  auto shape = getStaticShape(value);
  if (failed(shape))
    return failure();

  if (auto forOp = value.getDefiningOp<scf::ForOp>())
    return analyzeForResult(forOp, cast<OpResult>(value).getResultNumber());
  if (auto ifOp = value.getDefiningOp<scf::IfOp>())
    return analyzeIfResult(ifOp, cast<OpResult>(value).getResultNumber());

  if (!active.insert(value).second)
    return failure();
  auto guard = llvm::make_scope_exit([&] { active.erase(value); });

  auto facts = analyzeOperation(value);
  if (failed(facts))
    return failure();
  if (facts->invariant.size() != shape->size())
    return failure();
  cache.try_emplace(value, *facts);
  return *facts;
}

FailureOr<InvariantAxisAnalysis::AxisFacts>
InvariantAxisAnalysis::analyzeFactsOrVarying(Value value,
                                             ArrayRef<int64_t> shape) {
  auto facts = analyzeFacts(value);
  if (failed(facts) || facts->invariant.size() != shape.size())
    return makeFacts(shape.size(), false);
  return facts;
}

FailureOr<InvariantAxisAnalysis::AxisFacts>
InvariantAxisAnalysis::analyzeOperation(Value value) {
  auto shape = getStaticShape(value);
  if (failed(shape))
    return failure();
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return failure();

  if (isa<triton::SplatOp>(defOp))
    return makeFacts(shape->size(), true);

  if (auto constant = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
        dense && dense.isSplat())
      return makeFacts(shape->size(), true);
    return failure();
  }

  // Broadcast and expand_dims each manufacture one or more provably uniform
  // axes on their own. Only the axes inherited from the source need source
  // facts, so an unanalyzable source must not discard the newly created axes:
  // those axes stay `true` and the inherited ones degrade to `false`.
  if (auto broadcast = dyn_cast<triton::BroadcastOp>(defOp)) {
    auto sourceShape = getStaticShape(broadcast.getSrc());
    if (failed(sourceShape) || sourceShape->size() != shape->size())
      return failure();
    auto source = analyzeFactsOrVarying(broadcast.getSrc(), *sourceShape);
    if (failed(source))
      return failure();
    AxisFacts facts = makeFacts(shape->size(), true);
    for (auto [axis, sourceDim, resultDim] :
         llvm::enumerate(*sourceShape, *shape)) {
      if (sourceDim == resultDim) {
        facts.invariant[axis] = source->invariant[axis];
        continue;
      }
      // A size-one source dimension expands into a uniform result dimension.
      if (sourceDim != 1)
        return failure();
    }
    return facts;
  }

  if (auto expand = dyn_cast<triton::ExpandDimsOp>(defOp)) {
    auto sourceShape = getStaticShape(expand.getSrc());
    if (failed(sourceShape) || shape->size() != sourceShape->size() + 1)
      return failure();
    unsigned axis = expand.getAxis();
    if (axis >= shape->size())
      return failure();
    auto source = analyzeFactsOrVarying(expand.getSrc(), *sourceShape);
    if (failed(source))
      return failure();
    AxisFacts facts = makeFacts(shape->size(), true);
    for (unsigned i = 0, j = 0; i < shape->size(); ++i) {
      if (i == axis)
        continue;
      facts.invariant[i] = source->invariant[j++];
    }
    return facts;
  }

  if (auto reshape = dyn_cast<triton::ReshapeOp>(defOp)) {
    auto sourceShape = getStaticShape(reshape.getSrc());
    auto source = analyzeFacts(reshape.getSrc());
    if (failed(sourceShape) || failed(source))
      return failure();
    if (source->invariant.size() != sourceShape->size())
      return failure();
    // Reshape can reorder elements, so only an entirely uniform source is
    // guaranteed to stay uniform.
    if (llvm::all_of(source->invariant, [](bool bit) { return bit; }))
      return makeFacts(shape->size(), true);
    return failure();
  }

  if (auto bitcast = dyn_cast<triton::BitcastOp>(defOp)) {
    auto sourceShape = getStaticShape(bitcast.getSrc());
    auto source = analyzeFacts(bitcast.getSrc());
    if (failed(sourceShape) || failed(source))
      return failure();
    if (*sourceShape != *shape)
      return failure();
    // Pointer element-size changes do not affect which dimensions vary.
    if (isPointerTensor(bitcast.getSrc()) != isPointerTensor(value))
      return failure();
    return *source;
  }

  if (auto slice = dyn_cast<tensor::ExtractSliceOp>(defOp))
    return analyzeExtractSlice(slice, *shape);

  if (isa<triton::MakeRangeOp>(defOp))
    return makeFacts(shape->size(), false);

  // addptr: the pointer operand drives variation; a varying offset makes the
  // matching axis vary too.
  if (auto addPtr = dyn_cast<triton::AddPtrOp>(defOp)) {
    auto ptrShape = getStaticShape(addPtr.getPtr());
    auto ptrFacts = analyzeFacts(addPtr.getPtr());
    if (failed(ptrShape) || failed(ptrFacts))
      return failure();
    if (ptrFacts->invariant.size() != ptrShape->size())
      return failure();
    SmallVector<AxisFacts> inputs{*ptrFacts};
    if (isa<RankedTensorType>(addPtr.getOffset().getType())) {
      // An offset we cannot reason about may vary along any axis, so the
      // whole addptr must be treated as unanalyzable rather than silently
      // dropping the offset.
      auto offsetFacts = analyzeFacts(addPtr.getOffset());
      if (failed(offsetFacts))
        return failure();
      inputs.push_back(*offsetFacts);
    }
    return joinFacts(inputs, shape->size());
  }

  // Generic elementwise arithmetic: intersect the operand variation facts.
  if (isElementwiseArithmetic(defOp)) {
    SmallVector<AxisFacts> inputs;
    for (Value operand : defOp->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      auto operandShape = getStaticShape(operand);
      auto operandFacts = analyzeFacts(operand);
      if (failed(operandShape) || failed(operandFacts))
        return failure();
      if (*operandShape != *shape)
        return failure();
      inputs.push_back(*operandFacts);
    }
    if (inputs.empty())
      return makeFacts(shape->size(), true);
    return joinFacts(inputs, shape->size());
  }

  return failure();
}

FailureOr<InvariantAxisAnalysis::AxisFacts>
InvariantAxisAnalysis::analyzeExtractSlice(tensor::ExtractSliceOp slice,
                                           ArrayRef<int64_t> shape) {
  auto sourceShape = getStaticShape(slice.getSource());
  auto source = analyzeFacts(slice.getSource());
  if (failed(sourceShape) || failed(source))
    return failure();
  if (source->invariant.size() != sourceShape->size())
    return failure();

  // Only rank-preserving slices are handled here; a rank-reducing slice drops
  // dimensions and the axis correspondence would no longer hold.
  if (sourceShape->size() != shape.size())
    return failure();

  auto staticSizes = slice.getStaticSizes();
  auto staticStrides = slice.getStaticStrides();
  if (llvm::any_of(staticSizes, ShapedType::isDynamic) ||
      llvm::any_of(staticStrides, ShapedType::isDynamic))
    return failure();
  // A strided slice samples every k-th element and could break invariance.
  if (llvm::any_of(staticStrides, [](int64_t stride) { return stride != 1; }))
    return failure();

  AxisFacts facts = makeFacts(shape.size(), false);
  for (auto [axis, size] : llvm::enumerate(staticSizes)) {
    if (size == 1) {
      facts.invariant[axis] = true;
      continue;
    }
    if (static_cast<int64_t>(shape[axis]) == size)
      facts.invariant[axis] = source->invariant[axis];
  }
  return facts;
}

FailureOr<InvariantAxisAnalysis::AxisFacts>
InvariantAxisAnalysis::analyzeForIterArg(scf::ForOp forOp, unsigned index) {
  if (index >= forOp.getRegionIterArgs().size())
    return failure();
  auto init = analyzeFacts(forOp.getInitArgs()[index]);
  if (failed(init))
    return failure();

  BlockArgument iterArg = forOp.getRegionIterArgs()[index];
  AxisFacts candidate = *init;
  auto saved = bindings.find(iterArg);
  bool hadSaved = saved != bindings.end();
  AxisFacts savedFacts;
  if (hadSaved)
    savedFacts = saved->second;
  auto restore = llvm::make_scope_exit([&] {
    if (hadSaved)
      bindings[iterArg] = std::move(savedFacts);
    else
      bindings.erase(iterArg);
  });

  // Only invariant bits can be lost, so this fixed point terminates in at most
  // rank iterations.
  while (true) {
    bindings.insert_or_assign(iterArg, candidate);
    auto yielded = analyzeFacts(forOp.getYieldedValues()[index]);
    if (failed(yielded))
      return failure();
    if (yielded->invariant.size() != candidate.invariant.size())
      return failure();
    AxisFacts joined = candidate;
    for (auto [slot, bit] : llvm::zip(joined.invariant, yielded->invariant))
      slot = slot && bit;
    if (joined.invariant == candidate.invariant)
      break;
    candidate = std::move(joined);
    // Body expressions were analyzed under the previous assumption.
    cache.clear();
  }

  bindings.insert_or_assign(iterArg, candidate);
  restore.release();
  return candidate;
}

FailureOr<InvariantAxisAnalysis::AxisFacts>
InvariantAxisAnalysis::analyzeForResult(scf::ForOp forOp,
                                        unsigned resultIndex) {
  return analyzeForIterArg(forOp, resultIndex);
}

FailureOr<InvariantAxisAnalysis::AxisFacts>
InvariantAxisAnalysis::analyzeIfResult(scf::IfOp ifOp, unsigned resultIndex) {
  if (!ifOp.getElseRegion().hasOneBlock() ||
      resultIndex >= ifOp.getNumResults())
    return failure();
  auto thenFacts = analyzeFacts(ifOp.thenYield().getOperand(resultIndex));
  auto elseFacts = analyzeFacts(ifOp.elseYield().getOperand(resultIndex));
  if (failed(thenFacts) || failed(elseFacts))
    return failure();
  if (thenFacts->invariant.size() != elseFacts->invariant.size())
    return failure();
  AxisFacts joined = *thenFacts;
  for (auto [slot, bit] : llvm::zip(joined.invariant, elseFacts->invariant))
    slot = slot && bit;
  return joined;
}

} // namespace TritonToStructured