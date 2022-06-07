import numpy as np
from net import SimpleNet
import torch
import onnx
import onnx_graphsurgeon as gs
import os
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit

class Reproduce(self):

    def __init__(self):
        self.net = SimpleNet()
        if os.path.exists('./net.torch') is False:
            torch.save(self.net, './net.torch')
        
        self.rel_pos = np.load('./data/rel_pos.npy')
        self.hidden_states = np.load('./data/hidden_states.npy')

    def export_onnx(self):
        with torch.no_grad():
            torch.onnx.export(
            self.net,
            args=(self.rel_pos, self.hidden_states),
            f='./net.onnx',
            input_names=['rel_pos', 'hidden_states'],
            output_names=['out'],
            opset_version=11
            )

    def __gather_replace_onehot__(self):
        model = onnx.load("./net.onnx")
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
        onnx.save(gs.export_onnx(graph), "./new_net.onnx")

    def export_trt(self):
        self.__gather_replace_onehot__()
        os.system('trtexec --onnx=./new_net.onnx --saveEngine=net.plan --verbose --noTF32')


    def infer(self):
        # export onnx and trt
        if os.path.exists('./net.onnx') is False:
            self.export_onnx()
            self.export_trt()

        # torch out
        rel_pos = torch.from_numpy(self.rel_pos)
        hidden_states = torch.from_numpy(self.hidden_states)
        torch_out = self.SimpleNet(rel_pos, hidden_states)
        torch_out = torch_out.detach().numpy()

        # onnx out
        sess = onnxruntime.InferenceSession('./net.onnx')
        onnx_out = sess.run([], {'rel_pos':self.rel_pos})
        onnx_out = onnx_out[0]

        # new onnx out
        sess = onnxruntime.InferenceSession('./new_net.onnx')
        new_onnx_out = sess.run([], {'rel_pos':self.rel_pos})
        new_onnx_out = new_onnx_out[0]

        # trt out
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
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
        print('torch and onnx diff begin\n')
        result = 'ok'
        try:
            np.testing.assert_almost_equal(out, onnx_out, 5)
        except Exception as e:
            result = e
        print(result)
        print('torch and onnx diff end\n')

        print('onnx and new onnx diff begin\n')
        result = 'ok'
        try:
            np.testing.assert_almost_equal(onnx_out, new_onnx_out, 5)
        except Exception as e:
            result = e
        print(result)
        print('onnx and new onnx diff end\n')

        print('new onnx and trt diff begin\n')
        result = 'ok'
        try:
            np.testing.assert_almost_equal(new_onnx_out, trt_out, 5)
        except Exception as e:
            result = e
        print(result)
        print('new onnx and trt diff end\n')

if __name__ == '__main__':
    reproduce = Reproduce()
    reproduce.infer()
    

        





