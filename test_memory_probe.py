# To use, make sure memory_probe_ops.so is in $PATH or current directory

import tensorflow as tf
import numpy as np
import os
import sys

so_name = "memory_probe_ops.so"

paths_to_search = os.environ['PATH'].split(':')
paths_to_search.append('.')
for path in paths_to_search:
    full_so_name = os.path.join(path, so_name)
    if os.path.exists(full_so_name):
        print("Loading %s" %(full_so_name))
        memory_probe_ops = tf.load_op_library(full_so_name)
        break
else:
    print("Unable to find %s in . or $PATH, quitting"%(so_name))
    sys.exit()
    
run_with_tracing = False
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

for d in ["/cpu:0", "/gpu:0"]:
    with tf.device(d):
        sess = tf.InteractiveSession(config=config)
        mbs = 42
        n = mbs*250000
        inputs = tf.random_uniform((n,))
        print("Allocating %d MB variable on %s"%(mbs, d,))
        var = tf.Variable(inputs)
        probe_op = memory_probe_ops.bytes_in_use()
        max_op = memory_probe_ops.bytes_limit()
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        print("Before init %10d out of %10d bytes" % tuple(sess.run([probe_op, max_op])))
        sess.run(var.initializer)
        print("After  init %10d out of %10d bytes" % tuple(sess.run([probe_op, max_op])))
        if run_with_tracing:
            print(run_metadata)
