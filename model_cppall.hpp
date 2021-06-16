#ifndef MODEL_CPPALL_H
#define MODEL_CPPALL_H
#include <torch/torch.h>

struct EBImpl : torch::nn::Module
{
  EBImpl(int C_in, int C_out);

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Conv2d conv0;
  torch::nn::Conv2d conv1;
};
TORCH_MODULE(EB);

struct DBImpl : torch::nn::Module
{
  DBImpl(int C_in, int C_out);

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Conv2d conv0;
  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
};
TORCH_MODULE(DB);

struct EncoderImpl : torch::nn::Module
{
  EncoderImpl();

  std::vector<torch::Tensor> forward(torch::Tensor x);

  EB eb0;
  EB eb1;
  EB eb2;
  EB eb3;
  EB eb4;
  torch::nn::AvgPool2d avgpool;
};
TORCH_MODULE(Encoder);


struct DecoderImpl : torch::nn::Module
{
  DecoderImpl();

  torch::Tensor forward(std::vector<torch::Tensor> &cc4, std::vector<std::vector<int64_t>> &bsizes);

  DB db4;
  DB db3;
  DB db2;
  DB db1;
  DB db0;
  torch::nn::functional::InterpolateFuncOptions intOpts = torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>{2, 2});//.recompute_scale_factor(true);
};
TORCH_MODULE(Decoder);

struct OutnXImpl : torch::nn::Module
{
  OutnXImpl();

  torch::Tensor forward(torch::Tensor x, torch::Tensor &inputFrame);

  torch::nn::Conv2d conv0;
  torch::nn::Conv2d conv1;
};
TORCH_MODULE(OutnX);

struct UBnXImpl : torch::nn::Module
{
  UBnXImpl();

  torch::Tensor forward(torch::Tensor x);

  torch::nn::Conv2d conv0;
  torch::nn::Conv2d conv1;
};
TORCH_MODULE(UBnX);

struct SRModelImpl : torch::nn::Module
{
  SRModelImpl();

  void load_frame(torch::Tensor inputFrame1, torch::Tensor inputFrame2);

  torch::Tensor forward(torch::Tensor x);

  Encoder encoder;
  Decoder decoder;
  OutnX out1X;
  OutnX out2X;
  UBnX ub2X;
  std::vector<std::vector<torch::Tensor>> ecs;
  //std::vector<torch::Tensor> cc4;
  torch::Tensor cFrame;
  torch::Tensor nFrame;
  int i;
  std::vector<torch::Tensor> cc4;
  std::vector<std::vector<int64_t>> bsizes;
  //std::vector<double> scale = {2, 2};
  torch::nn::functional::InterpolateFuncOptions intOpts = torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>{2,2});//.recompute_scale_factor(true);
};
TORCH_MODULE(SRModel);


#endif // MODEL_CPPALL_H
