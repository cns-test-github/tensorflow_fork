/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TF_KERNEL_GEN_PASSES
#define TF_KERNEL_GEN_PASSES

include "mlir/Pass/PassBase.td"

def TFKernelToLLVMPass : Pass<"tf-kernel-to-llvm", "ModuleOp"> {
  let summary = "Pass for applying LLVM legalization patterns.";
  let constructor = "transforms::CreateTFKernelToLLVMPass()";
  let options = [
      Option<"blob_annotation_", "blob-annotation", "std::string",
           /*default=*/"\"gpu.binary_blob\"", "Blob attribute name">,
  ];
}

def EmbedTFFrameworkPass
    : Pass<"embed-tf-framework", "ModuleOp"> {
  let summary = "Pass to embed TF Framework for allocation and assertions,";
  let constructor = "tf_framework::CreateEmbedTFFrameworkPass()";
}

def RewriteTFFrameworkAssert : Pass<"rewrite-tf-framework-assert", "ModuleOp"> {
  let dependentDialects = ["cf::ControlFlowDialect"];
  let summary = "Pass to rewrite TFAssertOps to CondBranchOp";
  let constructor = "tf_framework::CreateRewriteTFFrameworkAssert()";
}

def FuncToJITInvocationPass : Pass<"func-to-jit-invocation", "mlir::func::FuncOp"> {
  let dependentDialects = [
      "mlir::kernel_gen::tf_framework::TFFrameworkDialect",
      "scf::SCFDialect",
      "shape::ShapeDialect"
  ];
  let summary = "Pass to rewrite a function to JIT invocations";
  let constructor = "transforms::CreateFuncToJITInvocationPass()";
  let options = [
      ListOption<"tile_sizes_", "tile-sizes", "int64_t", "Tiling sizes",
                 "llvm::cl::ZeroOrMore">,
      ListOption<"unroll_factors_", "unroll-factors", "int64_t",
                 "Unrolling in each tile dimension", "llvm::cl::ZeroOrMore">,
      Option<"enable_ftz_", "enable-ftz", "bool", /*default=*/"",
             "Enable the denormal flush to zero mode when generating code">,
      Option<"index_64bit_", "index_64bit", "bool", /*default=*/"",
             "Enable the 64 bit indexing for GPU kernels">,
      Option<"cpu_codegen_", "cpu-codegen", "bool", /*default=*/"",
             "CPU codegen (false implies GPU)">,
      Option<"jit_i64_indexed_for_large_tensors_",
             "jit_i64_indexed_for_large_tensors", "bool", /*default=*/"false",
             "Enable JIT compilation of i64-indexed kernels for large input "
             "tensors.">,
  ];
}

def BufferReusePass : Pass<"buffer-reuse", "mlir::func::FuncOp"> {
  let summary = "Pass to find and annotate candidates for buffer reuse.";
  let constructor = "transforms::CreateBufferReusePass()";
}

def ShapeToDescriptorsPass : Pass<"shape-to-descriptors", "ModuleOp"> {
  let summary = "Pass to transform shape computations to descriptors";
  let constructor = "transforms::CreateShapeToDescriptorsPass()";
}

def KernelgenFinalBufferizePass : Pass<"kernelgen-final-bufferize", "ModuleOp"> {
  let summary = "Pass to transform late operations on values to buffer based "
                "ones.";
  let constructor = "transforms::CreateKernelgenFinalBufferizePass()";
}

def GpuKernelToBlobPass : Pass<"gpu-kernel-to-blob", "gpu::GPUModuleOp"> {
  let summary = "Pass to annotate GPU Module with its PTX";
  let options = [
    Option<"blob_annotation_", "blob-annotation", "std::string",
           /*default=*/"\"gpu.binary_blob\"", "Blob attribute name">,
    ListOption<"architectures_", "arch", "std::string", "GPU architectures">,
    Option<"generate_fatbin_", "generate-fatbin", "bool", /*default=*/"true",
           "Bundle machine code for the different architectures in one "
           "fatbin.">,
    Option<"print_ptx_", "print-ptx", "bool", /*default=*/"false",
           "Print generated PTX code per target architecture.">,
    Option<"print_llvmir_", "print-llvmir", "bool", /*default=*/"false",
           "Print llvm ir when lowering code per target architecture.">,
  ];
  let constructor = "transforms::CreateGpuKernelToBlobPass()";
}

def ParallelLoopsToSequential : Pass<"parallel-loops-to-sequential", "mlir::func::FuncOp"> {
  let summary = "Pass to convert scf::ParallelOp to scf::ForOp";
  let constructor = "transforms::CreateParallelLoopsToSequential()";
}

def PropagateTfAbiKnowledgeToKernels
    : Pass<"propagate-tf-abi-knowledge-to-kernels", "mlir::func::FuncOp"> {
  let summary = "Pass to propagate tensorflow ABI knowledge to kernels";
  let constructor = "transforms::CreatePropagateTfAbiKnowledgeToKernels()";
}

def PropagateShapeKnowledgeToKernels
    : Pass<"propagate-shape-knowledge-to-kernels", "mlir::func::FuncOp"> {
  let summary = "Pass to propagate shape information into kernels";
  let constructor = "transforms::CreatePropagateShapeKnowledgeToKernels()";
}

def FuseInnerParallelLoopsPass
    : Pass<"fuse-inner-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Limited pass to forward stores to loads.";
  let constructor = "transforms::CreateFuseInnerParallelLoopsPass()";
  let description = [{
    Directs parallel loop fusion to the inner loops. This cannot be done with
    a passmanager alone ATM, as nested pass managers require operations to
    be closed from above.
  }];
}

def CopyCleanupPass : Pass<"copy-cleanup", "mlir::func::FuncOp"> {
  let summary = "Pass to remove copies which are consumed by a GenericOp.";
  let constructor = "transforms::CreateCopyCleanupPass()";
  let description = [{
    We can have GenericOps which have operands that are a copy, but the copy is
    not used by any other op. In this case, the GenericOp can just use the
    buffer which is the source of the copy, and we can remove the Alloc and the
    copy.
  }];
}

def ShapeSimplificationPass
    : Pass<"shape-simplification", "mlir::func::FuncOp"> {
  let summary = "Simplify shape ops";
}

def MergeAssumingOpsPass : Pass<"mhlo-merge-assuming-ops", "func::FuncOp"> {
  let summary = "Prepare moving dynamic broadcasts up over element-wise "
    "operations and broadcast the operands rather than the result. This will "
    "eventually allow for larger fusions.";
}

def BroadcastPropagationPass : Pass<"mhlo-broadcast-propagation", "func::FuncOp"> {
  let summary = "Move dynamic broadcasts up over element-wise operations and "
    "broadcast the operands rather than the result. This will eventually allow "
    "for larger fusions.";
}


#endif // TF_KERNEL_GEN_PASSES
