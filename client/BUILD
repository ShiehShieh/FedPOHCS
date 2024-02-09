package(default_visibility = ["//visibility:public"])

py_binary(
    name = "client",
    srcs = [
        "client.py",
    ],
    deps = [
        "//model/rl/comp:svf",
        "//model/utils:filters",
        "//model/utils:utils",
        "//model/utils:vectorization",
    ],
)
