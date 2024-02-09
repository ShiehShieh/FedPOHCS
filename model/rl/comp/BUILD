package(default_visibility = ["//visibility:public"])

load("//bazel:cython_library.bzl", "pyx_library")
# load("@cython//Tools:rules.bzl", "pyx_library")

py_library(
    name = "cfn_cell",
    srcs = ["cfn_cell.py"],
)

py_library(
    name = "policy_evaluation",
    srcs = ["policy_evaluation.py"],
)

py_library(
    name = "state_visitation_frequency",
    srcs = ["state_visitation_frequency.py"],
    deps = [
        "//model/rl/comp:policy_evaluation",
    ],
)

pyx_library(
    name = "svf",
    srcs = [
        "svf.pyx",
        "declarations.pxd",
        "setup.py",
    ],
    # data = glob([
    #     "setup.py",
    # ]),
    visibility = ["//visibility:public"],
    cython_options=[
        ('language', 'c++'),
        # ('extra_compile_args', '-O3'),
    ],
)
