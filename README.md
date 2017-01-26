# memory_probe_ops
TensorFlow kernels for probing memory.

Usage:

1. Copy `linux.memory_probe_ops.so` or `macos.memory_probe_ops.so` locally and rename to `memory_probe_ops.so`
2. Make sure this file is in current directory, or add its location to `$PATH`
3. Run test_memory_probe.py

If you put `.so` file in current directory, you can use it from script as follows

```
memory_probe_ops = tf.load_op_library("./memory_prob_ops.so")
print("Memory usage: ")
print(sess.run(memory_probe_ops.bytes_in_use()))
```


# Troubleshooting

- Getting `tensorflow.python.framework.errors.NotFoundError: ./memory_probe_ops.so: invalid ELF header`

Make sure you got the correct version of `.so` (Linux vs Mac)


# Building

## On MacOS
```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -undefined dynamic_lookup -shared memory_probe_ops.cc -o memory_probe_ops.so -fPIC -I $TF_INC -O2

```

## On Linux
```
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
g++ -std=c++11 -shared memory_probe_ops.cc -o memory_probe_ops.so -fPIC -I $TF_INC -O2

```
## Using Bazel
Put BUILD file with the following under `tensorflow/core/user_ops/`

```
tf_custom_op_library(
    name = "memory_probe_ops.so",
    srcs = ["memory_probe_ops.cc"],
)
```
Now run

`bazel build --config=cuda --config=opt //tensorflow/core/user_ops:memory_probe_ops.so
`

The `.so` file is dropped under `bazel-bin/tensorflow/core/user_ops/memory_probe_ops.so`
