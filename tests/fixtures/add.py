import onnx
from onnx import helper, TensorProto

# Define input tensors (matrices A and B)
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [3, 2, 1])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3, 2, 1])

# Define output tensor (A + B)
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [3, 2, 1])

# Create the Add node
node = helper.make_node(
    'Add',
    inputs=['A', 'B'],
    outputs=['C']
)

# Create the graph
graph = helper.make_graph(
    nodes=[node],
    name='AddGraph',
    inputs=[A, B],
    outputs=[C]
)

# Create the model
model = helper.make_model(graph, producer_name='php-ort-testing')

# Save the model to file
onnx.save(model, 'add.onnx')