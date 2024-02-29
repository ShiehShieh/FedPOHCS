package(default_visibility = ["//visibility:public"])

py_library(
    name = "fedbase",
    srcs = [
        "fedbase.py",
    ],
    deps = [
        "//config:config",
        "//model/cs:csh",
        "//model/cs:gradnorm",
        "//model/cs:powd",
        "//model/cs:random",
        "//model/rl/agent:critic",
        "//model/rl/agent:vec_agent",
        "//model/rl/comp:svf",
        "//model/utils:vectorization",
    ],
)

py_library(
    name = "fedavg",
    srcs = [
        "fedavg.py",
    ],
    deps = [
        ":fedbase",
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)

py_library(
    name = "fedprox",
    srcs = [
        "fedprox.py",
    ],
    deps = [
        ":fedbase",
        "//model/rl/agent:critic",
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)

py_library(
    name = "fedtrpo",
    srcs = [
        "fedtrpo.py",
    ],
    deps = [
        ":fedbase",
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)

py_library(
    name = "fmarl",
    srcs = [
        "fmarl.py",
    ],
    deps = [
        ":fedbase",
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)

py_library(
    name = "fedsgd",
    srcs = [
        "fedsgd.py",
    ],
    deps = [
        ":fedbase",
        "//model/rl/agent:vec_agent",
        "//model/utils:vectorization",
    ],
)
