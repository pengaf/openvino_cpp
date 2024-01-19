#include <iostream>
#include <fstream>
#include <vector>
#include <openvino/c/openvino.h>
#include <openvino/openvino.hpp>

void testc()
{
	ov_status_e status;

	ov_core_t* core = NULL;
	status = ov_core_create(&core);
	ov_model_t* model = NULL;
	status = ov_core_read_model(core, "G:/model/test.xml", "G:/model/test.bin", &model);

	ov_compiled_model_t* compiled_model = NULL;
	status = ov_core_compile_model(core, model, "AUTO", 0, &compiled_model);

	//ov_core_compile_model_from_file(core, "G:/model/test.onnx", "AUTO", 0, &compiled_model);

	ov_infer_request_t* infer_request = NULL;
	status = ov_compiled_model_create_infer_request(compiled_model, &infer_request);

	float memory_ptr[3 * 11 * 11] = {};
	// Get input port for model with one input
	ov_output_const_port_t* input_port = NULL;
	status = ov_model_const_input(model, &input_port);
	// Get the input shape from input port
	ov_shape_t input_shape;
	status = ov_const_port_get_shape(input_port, &input_shape);
	// Get the the type of input
	ov_element_type_e input_type;
	status = ov_port_get_element_type(input_port, &input_type);
	// Create tensor from external memory
	ov_tensor_t* tensor = NULL;
	status = ov_tensor_create_from_host_ptr(input_type, input_shape, memory_ptr, &tensor);
	// Set input tensor for model with one input
	status = ov_infer_request_set_input_tensor(infer_request, tensor);

	status = ov_infer_request_start_async(infer_request);
	status = ov_infer_request_wait(infer_request);

	ov_tensor_t* output_tensor = NULL;
	ov_tensor_t* output_tensor1 = NULL;
	// Get output tensor by tensor index
	status = ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor);
	status = ov_infer_request_get_output_tensor_by_index(infer_request, 1, &output_tensor1);

	void* data;
	status = ov_tensor_data(output_tensor1, &data);

	ov_shape_free(&input_shape);
	ov_tensor_free(output_tensor);
	ov_output_const_port_free(input_port);
	ov_tensor_free(tensor);
	ov_infer_request_free(infer_request);
	ov_compiled_model_free(compiled_model);
	ov_model_free(model);
	ov_core_free(core);
}

void testcpp()
{
	ov::Core core;
	ov::CompiledModel compiled_model = core.compile_model("G:/model/test.xml", "CPU");
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// Get input port for model with one input
	auto input_port = compiled_model.input();
	// Create tensor from external memory
	//auto shape = input_port.get_shape();
	auto partial_shape = input_port.get_partial_shape();
	//auto shape = partial_shape.get_min_shape();

	const size_t batch_size = 8;
	ov::Shape input_shape = { batch_size,3,11,11 };
	std::array<float, batch_size * 3 * 11 * 11> input{};
	ov::Tensor input_tensor(input_port.get_element_type(), input_shape, input.data());

	auto policy_port = compiled_model.output(0);
	auto value_port = compiled_model.output(1);

	ov::Shape policy_shape = { batch_size,11*11 };
	std::array<float, batch_size * 11 * 11> policy{};
	ov::Tensor policy_tensor(policy_port.get_element_type(), policy_shape, policy.data());


	ov::Shape value_shape = { batch_size, 1 };
	std::array<float, batch_size*1 > value{};
	ov::Tensor value_tensor(value_port.get_element_type(), value_shape, value.data());


	// Set input tensor for model with one input
	infer_request.set_input_tensor(input_tensor);
	infer_request.set_output_tensor(0, policy_tensor);
	infer_request.set_output_tensor(1, value_tensor);

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	for (uint32_t i = 0; i < 14000; ++i)
	{
		infer_request.infer();
		//infer_request.start_async();
		//infer_request.wait();
	}

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::duration startDuration = end - start;
	double seconds = startDuration.count() * 0.000000001;

	printf("%f", seconds);
	// Get output tensor by tensor name
	//auto output = infer_request.get_tensor("tensor_name");
	ov::Tensor outTensor = infer_request.get_output_tensor(0);
	ov::Tensor outTensor1 = infer_request.get_output_tensor(1);

	const float *output_buffer = outTensor.data<const float>();
	const float *output_buffer1 = outTensor1.data<const float>();
	// output_buffer[] - accessing output tensor data}
}

int main() 
{
	testcpp();
	return 0;
}