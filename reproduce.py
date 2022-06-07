import numpy as np
from net.simple_net import SimpleNet
import torch
import onnx
import onnx_graphsurgeon as gs
import os
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit
import onnxruntime
import sys

class Reproduce(object):

    def __init__(self):
        if os.path.exists('./net.torch') is True:
            self.net = torch.load('./net.torch')
        else:
            self.net = SimpleNet()
            torch.save(self.net, './net.torch')
             
        self.rel_pos = np.load('./data/rel_pos.npy')
        self.hidden_states = np.load('./data/hidden_states.npy')

    def export_onnx(self):
        rel_pos = torch.from_numpy(self.rel_pos)
        hidden_states = torch.from_numpy(self.hidden_states)
        with torch.no_grad():
            torch.onnx.export(
            self.net,
            args=(rel_pos, hidden_states),
            f='./onehot_net.onnx',
            input_names=['rel_pos', 'hidden_states'],
            output_names=['out'],
            opset_version=11
            )

    def __gather_replace_onehot__(self):
        model = onnx.load("./onehot_net.onnx")
        graph = gs.import_onnx(model)
        for node in graph.nodes:
            if node.op == "OneHot":
                # onehot + MulAdd = gather
                indices = node.inputs[0]
                data = node.o().o().inputs[1]
                gather_out = gs.Variable(name="gather", dtype=np.float32, shape=None)
                gather_node = gs.Node(name="gather", op="Gather", inputs=[data, indices],
                                    outputs=node.o().o().outputs)
                graph.nodes.append(gather_node)
                node.inputs = []
                node.o().o().outputs = []
        graph.cleanup().toposort()
        onnx.save(gs.export_onnx(graph), "./gather_net.onnx")

    def export_trt(self):
        self.__gather_replace_onehot__()
        os.system('trtexec --onnx=./gather_net.onnx --saveEngine=net.plan --verbose --noTF32')


    def infer(self):
        # export onnx and trt
        if os.path.exists('./onhot_net.onnx') is False:
            self.export_onnx()
            self.export_trt()

        # torch out
        rel_pos = torch.from_numpy(self.rel_pos)
        hidden_states = torch.from_numpy(self.hidden_states)
        net = torch.load('./net.torch')
        torch_out = net(rel_pos, hidden_states)
        torch_out = torch_out.detach().numpy()

        # onnx out
        sess = onnxruntime.InferenceSession('./onehot_net.onnx')
        onehot_onnx_out = sess.run([], {'rel_pos':self.rel_pos})
        onehot_onnx_out = onehot_onnx_out[0]

        # new onnx out
        sess = onnxruntime.InferenceSession('./gather_net.onnx')
        gather_onnx_out = sess.run([], {'rel_pos':self.rel_pos})
        gather_onnx_out = gather_onnx_out[0]

        # trt out
        G_LOGGER = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(G_LOGGER)
        engine_file = './net.plan'
        
        with open(engine_file, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        inputs = [self.rel_pos]
        bindings = []
        for i in inputs:
            d_i = cuda.mem_alloc(i.size * i.dtype.itemsize)
            cuda.memcpy_htod(d_i, i)
            bindings.append(int(d_i))

        out_shape = context.get_binding_shape(1)
        trt_out = np.empty(out_shape, dtype=np.float32)
        d_o = cuda.mem_alloc(trt_out.size* trt_out.dtype.itemsize)
        bindings.append(int(d_o))
        stream = cuda.Stream()

        context.execute_v2(bindings)
        cuda.memcpy_dtoh(trt_out, d_o)

        # diff
        print('torch and onehot_onnx diff begin\n')
        result = 'ok\n'
        try:
            np.testing.assert_almost_equal(torch_out, onehot_onnx_out, 5)
        except Exception as e:
            result = e
        print(result)
        print('torch and onehot_onnx diff end\n')

        print('onehot_onnx and gather_onnx diff begin\n')
        result = 'ok\n'
        try:
            np.testing.assert_almost_equal(onehot_onnx_out, gather_onnx_out, 5)
        except Exception as e:
            result = e
        print(result)
        print('onehot_onnx and gather_onnx diff end\n')

        print('gather_onnx and trt diff begin\n')
        result = 'ok\n'
        try:
            np.testing.assert_almost_equal(gather_onnx_out, trt_out, 5)
        except Exception as e:
            result = e
        print(result)
        print('gather_onnx and trt diff end\n')

if __name__ == '__main__':
    if len(sys.argv) != 1:
        os.system('rm net.torch onehot_net.onnx')
        print('clean\n')
        sys.exit()
    reproduce = Reproduce()
    reproduce.infer()
    

        





