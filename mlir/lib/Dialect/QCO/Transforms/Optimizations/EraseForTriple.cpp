/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::qco {

#define GEN_PASS_DEF_ERASEFORTRIPLE
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

struct EraseForTriple final : impl::EraseForTripleBase<EraseForTriple> {
  using EraseForTripleBase::EraseForTripleBase;

protected:
  void runOnOperation() override {
    const auto op = getOperation();
    auto* ctx = &getContext();

    // TODO: Task 4 (Bonus)
    
    // PassManager pm(ctx);
    // pm.addNestedPass<func::FuncOp>(createQuantumLoopUnroll());
    // pm.addPass(getCanonicalizationPatterns(results, ctx));

    // Collect and unroll quantum loops with a constant trip count of 3.
    auto isQuantumLoop = [](scf::ForOp loop) {
      return llvm::any_of(loop.getInitArgs(), [](Value arg) {
        if (isa<QubitType>(arg.getType())) {
          return true;
        }
        if (const auto tensorTy = dyn_cast<RankedTensorType>(arg.getType())) {
          return isa<QubitType>(tensorTy.getElementType());
        }
        return false;
      });
    };

    auto collectQuantumLoops = [&](FunctionOpInterface func) {
      SmallVector<scf::ForOp> loops;
      func.walk<WalkOrder::PostOrder>([&](scf::ForOp loop) {
        if (isQuantumLoop(loop))
          loops.emplace_back(loop);
      });
      return loops;
    };

    auto module = getOperation();
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      for (auto loop : collectQuantumLoops(funcOp)) {
        auto tripCounts = mlir::getConstLoopTripCounts(loop);
        if (!tripCounts.empty() && tripCounts[0] == 3) {
          if (failed(loopUnrollFull(loop))) {
            loop.emitError() << "failed to fully unroll";
            signalPassFailure();
            return;
          }
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::qco
