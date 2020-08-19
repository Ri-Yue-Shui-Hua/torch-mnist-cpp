//#include <torch/torch.h>
//
//#include <cstddef>
//#include <iostream>
//#include <string>
//#include <vector>
//
//
//struct Net : torch::nn::Module {
//	Net()
//		: conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
//		conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
//		fc1(320, 50),
//		fc2(50, 10) {
//		register_module("conv1", conv1);
//		register_module("conv2", conv2);
//		register_module("conv2_drop", conv2_drop);
//		register_module("fc1", fc1);
//		register_module("fc2", fc2);
//	}
//
//	torch::Tensor forward(torch::Tensor x) {
//		x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
//		x = torch::relu(
//			torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
//		x = x.view({ -1, 320 });
//		x = torch::relu(fc1->forward(x));
//		x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
//		x = fc2->forward(x);
//		return torch::log_softmax(x, /*dim=*/1);
//	}
//
//	torch::nn::Conv2d conv1;
//	torch::nn::Conv2d conv2;
//	torch::nn::FeatureAlphaDropout conv2_drop;
//	torch::nn::Linear fc1;
//	torch::nn::Linear fc2;
//};
//
//struct Options {
//	std::string data_root{ "data" };
//	int32_t batch_size{ 64 };
//	int32_t epochs{ 10 };
//	double lr{ 0.01 };
//	double momentum{ 0.5 };
//	bool no_cuda{ false };
//	int32_t seed{ 1 };
//	int32_t test_batch_size{ 1000 };
//	int32_t log_interval{ 10 };
//};
//
//template <typename DataLoader>
//void train(
//	int32_t epoch,
//	const Options& options,
//	Net& model,
//	torch::Device device,
//	DataLoader& data_loader,
//	torch::optim::SGD& optimizer,
//	size_t dataset_size) {
//	model.train();
//	size_t batch_idx = 0;
//	for (auto& batch : data_loader) {
//		auto data = batch.data.to(device), targets = batch.target.to(device);
//		optimizer.zero_grad();
//		auto output = model.forward(data);
//		auto loss = torch::nll_loss(output, targets);
//		loss.backward();
//		optimizer.step();
//
//		if (batch_idx++ % options.log_interval == 0) {
//			std::cout << "Train Epoch: " << epoch << " ["
//				<< batch_idx * batch.data.size(0) << "/" << dataset_size
//				<< "]\tLoss: " << loss.template item<float>() << std::endl;
//		}
//	}
//}
//
//template <typename DataLoader>
//void test(
//	Net& model,
//	torch::Device device,
//	DataLoader& data_loader,
//	size_t dataset_size) {
//	torch::NoGradGuard no_grad;
//	model.eval();
//	double test_loss = 0;
//	int32_t correct = 0;
//	for (const auto& batch : data_loader) {
//		auto data = batch.data.to(device), targets = batch.target.to(device);
//		auto output = model.forward(data);
//		test_loss += torch::nll_loss(
//			output,
//			targets,
//			/*weight=*/{},
//			at::Reduction::Sum).item<double>();
//		auto pred = output.argmax(1);
//		correct += pred.eq(targets).sum().template item<int64_t>();
//	}
//
//	test_loss /= dataset_size;
//	std::cout << "Test set: Average loss: " << test_loss
//		<< ", Accuracy: " << correct << "/" << dataset_size << std::endl;
//}
//
//struct Normalize : public torch::data::transforms::TensorTransform<> {
//	Normalize(float mean, float stddev)
//		: mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
//	torch::Tensor operator()(torch::Tensor input) {
//		return input.sub_(mean_).div_(stddev_);
//	}
//	torch::Tensor mean_, stddev_;
//};
//
//auto main(int argc, const char* argv[]) -> int {
//	torch::manual_seed(0);
//
//	Options options;
//	torch::DeviceType device_type;
//	if (torch::cuda::is_available() && !options.no_cuda) {
//		std::cout << "CUDA available! Training on GPU" << std::endl;
//		device_type = torch::kCUDA;
//	}
//	else {
//		std::cout << "Training on CPU" << std::endl;
//		device_type = torch::kCPU;
//	}
//	torch::Device device(device_type);
//
//	Net model;
//	model.to(device);
//	
//	auto train_dataset =
//		torch::data::datasets::MNIST(
//			options.data_root, torch::data::datasets::MNIST::Mode::kTrain)
//		.map(Normalize(0.1307, 0.3081))
//		.map(torch::data::transforms::Stack<>());
//	const auto dataset_size = train_dataset.size();
//
//	auto train_loader = torch::data::make_data_loader(
//		std::move(train_dataset), options.batch_size);
//
//	auto test_loader = torch::data::make_data_loader(
//		torch::data::datasets::MNIST(
//			options.data_root, torch::data::datasets::MNIST::Mode::kTest)
//		.map(Normalize(0.1307, 0.3081))
//		.map(torch::data::transforms::Stack<>()),
//		options.batch_size);
//
//	torch::optim::SGD optimizer(
//		model.parameters(),
//		torch::optim::SGDOptions(options.lr).momentum(options.momentum));
//
//	for (size_t epoch = 1; epoch <= options.epochs; ++epoch) {
//		train(
//			epoch, options, model, device, *train_loader, optimizer, dataset_size.value());
//		test(model, device, *test_loader, dataset_size.value());
//	}
//}


#include "Lenet-5.h"


template <typename  DataLoader>
void train(std::shared_ptr<Lenet5> &model, DataLoader &loader, torch::optim::Adam &optimizer) {
	model->train();
	// 迭代数据
	int n = 0;
	for (torch::data::Example<torch::Tensor, torch::Tensor> &batch : loader) {
		torch::Tensor data = batch.data;
		auto target = batch.target;
		optimizer.zero_grad(); // 清空上一次的梯度
		// 计算预测值
		torch::Tensor y = model->forward(data);
		// 计算误差
		torch::Tensor loss = torch::nll_loss(y, target);
		// 计算梯度: 前馈求导
		loss.backward();
		// 根据梯度更新参数矩阵
		optimizer.step();
		// 为了观察效果，输出损失
		// std::cout << "\t|--批次：" << std::setw(2) << std::setfill(' ')<< ++n 
		//           << ",\t损失值：" << std::setw(8) << std::setprecision(4) << loss.item<float>() << std::endl;
	}

	// 输出误差
}
template <typename DataLoader>
void  valid(std::shared_ptr<Lenet5> &model, DataLoader &loader) {
	model->eval();
	// 禁止求导的图跟踪
	torch::NoGradGuard  no_grad;
	// 循环测试集
	double sum_loss = 0.0;
	int32_t num_correct = 0;
	int32_t num_samples = 0;
	for (const torch::data::Example<> &batch : loader) {
		// 每个批次预测值
		auto data = batch.data;
		auto target = batch.target;
		num_samples += data.sizes()[0];
		auto y = model->forward(data);
		// 计算纯预测的结果
		auto pred = y.argmax(1);
		// 计算损失值
		sum_loss += torch::nll_loss(y, target, {}, at::Reduction::Sum).item<double>();
		// 比较预测结果与真实的标签值
		num_correct += pred.eq(target).sum().item<int32_t>();
	}
	// 输出正确值
	std::cout << std::setw(8) << std::setprecision(4)
		<< "平均损失值：" << sum_loss / num_samples
		<< ",\t准确率：" << 100.0 * num_correct / num_samples << " %" << std::endl;
}

int main(int argc, const char** argv) {

	// 数据集
	auto  ds_train = torch::data::datasets::MNIST(".\\data", torch::data::datasets::MNIST::Mode::kTrain);
	auto  ds_valid = torch::data::datasets::MNIST(".\\data", torch::data::datasets::MNIST::Mode::kTest);

	// torch::data::transforms::Normalize<> norm(0.1307, 0.3081);
	torch::data::transforms::Stack<> stack;

	// 数据批次加载器
	// auto n_train = ds_train.map(norm);
	auto s_train = ds_train.map(stack);
	auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(s_train), 1000);

	// auto n_valid = ds_valid.map(norm);
	auto s_valid = ds_valid.map(stack);
	auto valid_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(s_valid), 1000);

	// 1. 创建模型对象
	std::shared_ptr<Lenet5> model = std::make_shared<Lenet5>();
	// for(auto &batch: *train_loader){
	//     auto data = batch.data;
	//     auto target = batch.target;
	//     data = data.view({-1, 1, 28, 28});
	//     auto pred = model->forward(data);
	//     // pred  <-> target 存在误差，计算误差，计算调整5 * 5 核矩阵的依据，调整的方向是 loss(pred - target) -> 0 
	// }


	// 优化器（管理模型中可训练矩阵）
	torch::optim::Adam  optimizer = torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(0.001)); // 根据经验一般设置为10e-4 

	std::cout << "开始训练" << std::endl;
	int epoch = 20;
	int interval = 1;   // 从测试间隔
	for (int e = 0; e < epoch; e++) {
		std::printf("第%02d论训练\n", e + 1);
		train(model, *train_loader, optimizer);
		if (e  % interval == 0) {
			valid(model, *valid_loader);
		}
	}
	std::cout << "训练结束" << std::endl;
	torch::save(model, "lenet5.pt");
	return 0;
}


