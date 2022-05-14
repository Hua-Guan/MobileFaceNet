import torch
from mobilefacenet import MobileFaceNet
torch_model = MobileFaceNet()  					# 由研究员提供python.py文件
torch_model.load_state_dict(torch.load('best.pt', map_location=torch.device('cpu')))
batch_size = 1 								# 批处理大小
input_shape = (3, 112, 112) 				# 输入数据

# set the model to inference mode
torch_model.eval()

x = torch.randn(batch_size,*input_shape) 	# 生成张量
export_onnx_file = "test.onnx" 				# 目的ONNX文件名
torch.onnx.export(torch_model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],		# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},	# 批处理变量
                                    "output":{0:"batch_size"}})

