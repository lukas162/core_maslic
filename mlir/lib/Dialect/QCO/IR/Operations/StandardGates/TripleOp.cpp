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

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

struct RemoveThreeBackToBackTripleOps final : OpRewritePattern<TripleOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TripleOp op,
                                PatternRewriter& rewriter) const override {
    // TODO: Task 3
    auto nextOp = dyn_cast<TripleOp>(*op.getOutputQubit(0).user_begin());
    if (!nextOp) {
      return failure();
    }

    auto nextnextOp = dyn_cast<TripleOp>(*nextOp.getOutputQubit(0).user_begin());
    if (!nextnextOp) {
      return failure();
    }

    // Erase all three operations
    rewriter.replaceOp(op, op.getInputQubits());
    rewriter.replaceOp(nextOp, nextOp.getInputQubits());
    rewriter.replaceOp(nextnextOp, nextnextOp.getInputQubits());
    return success();
    }
  }
};

} // namespace

void TripleOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveThreeBackToBackTripleOps>(context);
}
