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

#ifndef XLA_BACKENDS_AUTOTUNER_BACKENDS_CPU_XNNPACK_BACKEND_H_
#define XLA_BACKENDS_AUTOTUNER_BACKENDS_CPU_XNNPACK_BACKEND_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/autotuner/backends/cpu/cpu_codegen_backend.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/compiler.h"

namespace xla::cpu {

class XnnpackBackend : public CpuCodegenBackend {
 public:
  using Config = XnnFusionThunkProto::Options;

  XnnpackBackend() : CpuCodegenBackend("xnnpack") {}

  absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
  GetSupportedConfigs(const HloInstruction& instr) final;

  absl::StatusOr<std::unique_ptr<BackendConfig>> GetDefaultConfig(
      const HloInstruction& instr) final;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_AUTOTUNER_BACKENDS_CPU_XNNPACK_BACKEND_H_
