from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time


# See this page : http://deeplearning.net/software/theano/tutorial/using_gpu.html
# From console :
# $  THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python gpu_check.py
# $  THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python gpu_check.py


def test():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = numpy.random.RandomState(22)
    x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
    f = function([], T.exp(x))
    print(f.maker.fgraph.toposort())
    t0 = time.time()
    for i in range(iters):
        r = f()
    t1 = time.time()
    print("Looping %d times took %f seconds" % (iters, t1 - t0))
    print("Result is %s" % (r,))
    if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print('Used the cpu')
        return False
    else:
        print('Used the gpu')
        return True


if __name__ == "__main__":
    t = test()