load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

cc_library(
    name = "cpu_codegen_backend",
    hdrs = ["cpu_codegen_backend.h"],
    deps = [
        "//xla:xla_proto_cc",
        "//xla/backends/autotuner:codegen_backend",
        "//xla/hlo/ir:hlo",
        "//xla/service:compiler",
        "//xla/service:executable",
        "//xla/stream_executor:platform_manager",
        "//xla/stream_executor:stream_executor_h",
        "//xla/stream_executor/host:host_platform",
        "//xla/tools:hlo_decomposer_lib",
        "//xla/tsl/platform:status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
    alwayslink = 1,  # needed for platform registration
)
