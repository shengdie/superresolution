#include "model_cppall.hpp"
#include <chrono>
#include <cmath>

using namespace torch;

EBImpl::EBImpl(int C_in, int C_out)
    : conv0(nn::Conv2dOptions(C_in, C_out, 3).padding(1).stride(1)),
      conv1(nn::Conv2dOptions(C_out, C_out, 3).padding(1).stride(1))
{
    register_module("Conv3x3_0", conv0);
    register_module("Conv3x3_1", conv1);
}

torch::Tensor EBImpl::forward(torch::Tensor x)
{
    x = torch::leaky_relu(conv0->forward(x), 0.2);
    return torch::leaky_relu(conv1->forward(x), 0.2);
}

DBImpl::DBImpl(int C_in, int C_out) : conv0(nn::Conv2dOptions(C_in, C_out, 3).padding(1)),
                                      conv1(nn::Conv2dOptions(C_out, C_out, 3).padding(1)),
                                      conv2(nn::Conv2dOptions(C_out, C_out, 3).padding(1))
{
    register_module("Conv3x3_0", conv0);
    register_module("Conv3x3_1", conv1);
    register_module("Conv3x3_2", conv2);
}

torch::Tensor DBImpl::forward(torch::Tensor x)
{
    x = torch::leaky_relu(conv0->forward(x), 0.2);
    x = torch::leaky_relu(conv1->forward(x), 0.2);
    return torch::leaky_relu(conv2->forward(x), 0.2);
}

EncoderImpl::EncoderImpl() : eb0(3, 48),
                             eb1(48, 80),
                             eb2(80, 144),
                             eb3(144, 256),
                             eb4(256, 384),
                             avgpool(nn::AvgPool2dOptions(2).stride(2).ceil_mode(true).count_include_pad(false))
{
    register_module("EB0", eb0);
    register_module("EB1", eb1);
    register_module("EB2", eb2);
    register_module("EB3", eb3);
    register_module("EB4", eb4);
    register_module("Avgpool", avgpool);
}

std::vector<torch::Tensor> EncoderImpl::forward(torch::Tensor x)
{
    //std::vector<torch::Tensor> vec;
    torch::Tensor ebr0 = eb0->forward(x);
    x = avgpool->forward(ebr0);
    auto ebr1 = eb1->forward(x);
    x = avgpool->forward(ebr1);
    auto ebr2 = eb2->forward(x);
    x = avgpool->forward(ebr2);
    auto ebr3 = eb3->forward(x);
    x = avgpool->forward(ebr3);
    auto ebr4 = eb4->forward(x);
    return {ebr0, ebr1, ebr2, ebr3, ebr4};
}

DecoderImpl::DecoderImpl() : db4(1920, 384),
                             db3(1664, 384),
                             db2(1104, 240),
                             db1(640, 136),
                             db0(376, 80)
{
    register_module("DB4", db4);
    register_module("DB3", db3);
    register_module("DB2", db2);
    register_module("DB1", db1);
    register_module("DB0", db0);
}

torch::Tensor DecoderImpl::forward(std::vector<torch::Tensor> &cc4, std::vector<std::vector<int64_t>> &bsizes)
{
    torch::Tensor x;

    x = nn::functional::interpolate(db4->forward(cc4[0]), nn::functional::InterpolateFuncOptions().size(bsizes[3]));
    x = torch::cat({x, cc4[1]}, 1);
    x = nn::functional::interpolate(db3->forward(x),  nn::functional::InterpolateFuncOptions().size(bsizes[2]));
    x = torch::cat({x, cc4[2]}, 1);
    x = nn::functional::interpolate(db2->forward(x),  nn::functional::InterpolateFuncOptions().size(bsizes[1]));
    x = torch::cat({x, cc4[3]}, 1);
    x = nn::functional::interpolate(db1->forward(x),  nn::functional::InterpolateFuncOptions().size(bsizes[0]));
    x = torch::cat({x, cc4[4]}, 1);
    return db0->forward(x);
}

OutnXImpl::OutnXImpl() : conv0(nn::Conv2dOptions(80, 40, 3).padding(1)),
                         conv1(nn::Conv2dOptions(40, 3, 3).padding(1))
{
    register_module("Conv3x3_0", conv0);
    register_module("Conv3x3_o", conv1);
}

torch::Tensor OutnXImpl::forward(torch::Tensor x, torch::Tensor &inputFrame)
{
    x = torch::leaky_relu(conv0->forward(x), 0.2);
    return conv1->forward(x) + inputFrame;
}

UBnXImpl::UBnXImpl() : conv0(nn::Conv2dOptions(80, 80, 3).padding(1)),
                       conv1(nn::Conv2dOptions(80, 80, 3).padding(1))
{
    register_module("Conv3x3_0", conv0);
    register_module("Conv3x3_1", conv1);
}

torch::Tensor UBnXImpl::forward(torch::Tensor x)
{
    x = torch::leaky_relu(conv0->forward(x), 0.2);
    return torch::leaky_relu(conv1->forward(x), 0.2);
}

SRModelImpl::SRModelImpl() : encoder(), decoder(), out1X(), out2X(), ub2X()
{
  register_module("Encoder", encoder);
  register_module("Decoder", decoder);
  register_module("Out1X", out1X);
  register_module("Out2X", out2X);
  register_module("UB2X", ub2X);
  i = 0;
}

void SRModelImpl::load_frame(torch::Tensor inputFrame1, torch::Tensor inputFrame2)
{
  auto x = encoder->forward(inputFrame1);
  // std::vector<torch::Tensor> x1(x);
  // std::vector<torch::Tensor> x2(x);
  // std::vector<torch::Tensor> x3(x);
  ecs.push_back(x);
  ecs.push_back(x);
  ecs.push_back(x);
  ecs.push_back(encoder->forward(inputFrame2));
  ecs.push_back(x);
  cFrame = inputFrame1.clone();
  nFrame = inputFrame2.clone();
  int64_t h0, w0;
  h0 = cFrame.size(2);
  w0 = cFrame.size(3);
  for(int j =0; j <4; j++){
    bsizes.push_back({h0, w0});
    h0 = (int64_t)std::ceil((h0 - 2)/2.0 + 1);
    w0 = (int64_t)std::ceil((w0 - 2)/2.0 + 1);
  }

  // for (int j = 4; j >= 0; j--)
  //   cc4.emplace_back(torch::cat({ecs[(0 + i) % 5][j], ecs[(1 + i) % 5][j], ecs[(2 + i) % 5][j], ecs[(3 + i) % 5][j], ecs[(4 + i) % 5][j]}, 1));
}

torch::Tensor SRModelImpl::forward(torch::Tensor x)
{
//   std::chrono::steady_clock::time_point begin; // = std::chrono::steady_clock::now();
//   std::chrono::steady_clock::time_point end;
//   std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();
  ecs.at((4 + i) % 5) = encoder->forward(x);
  //end = std::chrono::steady_clock::now();
  //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin0).count() << "[µs]" << std::endl;

  cc4.clear();
  //begin = std::chrono::steady_clock::now();
  for (int j = 4; j >= 0; j--)
   cc4.emplace_back(torch::cat({ecs[(0 + i) % 5][j], ecs[(1 + i) % 5][j], ecs[(2 + i) % 5][j], ecs[(3 + i) % 5][j], ecs[(4 + i) % 5][j]}, 1));
  
  // int64_t siz;
  // auto ec = encoder->forward(x);
  // for (int j = 4; j >= 0; j--) {
  //   siz = ec[j].size(1);
  //   cc4[j].slice(1, 0, siz) = ec[j];
  //   cc4.at(j) = cc4[j].roll(-siz, 1);
  //   //cc4[j][0].slice()
  // }
//   end = std::chrono::steady_clock::now(); 
//   std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
  
//   begin = std::chrono::steady_clock::now();
  auto decs = decoder->forward(cc4, bsizes);
//   end = std::chrono::steady_clock::now(); 
//   std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

  auto o1x = out1X->forward(decs, cFrame);
  auto dec2x = nn::functional::interpolate(decs, intOpts);
  auto o1x2x = nn::functional::interpolate(o1x, intOpts);
  dec2x = ub2X->forward(dec2x);
  auto o2x = out2X->forward(dec2x, o1x2x);
  cFrame = nFrame;
  nFrame = x;
  i = (i+1) % 5;
//   end = std::chrono::steady_clock::now(); 
//   std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin0).count() << "[µs]" << std::endl;
  return o2x;
}