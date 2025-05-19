/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_AUTOTUNER_BACKENDS_CPU_CPU_CODEGEN_BACKEND_H_
#define XLA_BACKENDS_AUTOTUNER_BACKENDS_CPU_CPU_CODEGEN_BACKEND_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tsl/platform/status.h"
#include "xla/xla.pb.h"

namespace xla {
namespace cpu {

// Abstract base class for CPU backends, implementing the CodegenBackend
// interface.
class CpuCodegenBackend : public CodegenBackend {
 public:
  explicit CpuCodegenBackend(absl::string_view name) : name_(name) {
    auto platform_or_err =
        stream_executor::PlatformManager::PlatformWithName("host", true);
    TF_CHECK_OK(platform_or_err.status());
    auto compiler_or_err = Compiler::GetForPlatform(platform_or_err.value());
    TF_CHECK_OK(compiler_or_err.status());
    compiler_ = std::move(compiler_or_err.value());
  }

  absl::string_view name() const override { return name_; }

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr,
                      stream_executor::StreamExecutor* stream_executor) final {
    return GetSupportedConfigs(instr);
  }

  // No need for stream executor for CPU backends.
  virtual absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) = 0;

  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& hlo_instruction,
      const BackendConfig& config) override {
    std::unique_ptr<HloModule> hlo_module =
        ExtractInstructionIntoNewModule(hlo_instruction);

    Compiler::CompileOptions options;
    options.is_autotuning_compilation = true;

    return compiler_->RunBackend(std::move(hlo_module),
                                 /*executor=*/nullptr, options);
  }

 private:
  std::string name_;
  std::unique_ptr<Compiler> compiler_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_BACKENDS_CPU_CPU_CODEGEN_BACKEND_H_
