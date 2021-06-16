#include "state_loader.hpp"
#include "model_cppall.hpp"

//#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <regex>
#include <stack>
#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>
// #include <opencv2/core/core.hpp>
// #include <opencv2/videoio.hpp>
// #include <opencv2/imgcodecs.hpp>

using namespace torch;
//using namespace cv;

// void mat2tensor(cv::Mat &input, at::Tensor &t){
//     //auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
//     //t = torch::from_blob(input.ptr<float>(), /*sizes=*/{1, input.rows, input.cols, 3}, options);
//     //float* tensorData = frameTensor.data_ptr<float>();
//     cv::Mat outputImg(input.rows,  input.cols, CV_32FC3, t.data_ptr<float>());
//     input.convertTo(outputImg, CV_32FC3, 1.0f / 255.0f);
//     //outputImg = outputImg/255.0f;
//     //t.permute({0, 3, 1, 2});
// }

// void tensor2mat(at::Tensor &t, cv::Mat &image){
//     //t.permute({0, 2, 3, 1});
//     image = cv::Mat(t.size(1), t.size(2), CV_8UC3, t.data_ptr<u_int8_t>());
//     //image *= 255.;
// 	//image.convertTo(image, CV_8UC3);
// }

struct EBImpl : nn::Module
{
  EBImpl(int C_in, int C_out)
      : conv0(nn::Conv2dOptions(C_in, C_out, 3).padding(1).stride(1)),
        //lrelu0(nn::LeakyReLUOptions().negative_slope(0.2).inplace(false)),
        conv1(nn::Conv2dOptions(C_out, C_out, 3).padding(1).stride(1)) //,
  //lrelu1(nn::LeakyReLUOptions().negative_slope(0.2).inplace(false))
  {
    register_module("Conv3x3_0", conv0);
    //register_module("Lrelu0", lrelu0);
    register_module("Conv3x3_1", conv1);
    //register_module("Lrelu1", lrelu1);
    //register_module("Conv")
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::leaky_relu(conv0->forward(x), 0.2);
    return torch::leaky_relu(conv1->forward(x), 0.2);
  }

  //torch::nn::Conv2d
  nn::Conv2d conv0;
  nn::Conv2d conv1;
  //nn::LeakyReLU lrelu0;
  //nn::LeakyReLU lrelu1;
};
TORCH_MODULE(EB);

struct DBImpl : nn::Module
{
  DBImpl(int C_in, int C_out) : conv0(nn::Conv2dOptions(C_in, C_out, 3).padding(1)),
                                conv1(nn::Conv2dOptions(C_out, C_out, 3).padding(1)),
                                conv2(nn::Conv2dOptions(C_out, C_out, 3).padding(1))
  {
    register_module("Conv3x3_0", conv0);
    register_module("Conv3x3_1", conv1);
    register_module("Conv3x3_2", conv2);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::leaky_relu(conv0->forward(x), 0.2);
    x = torch::leaky_relu(conv1->forward(x), 0.2);
    return torch::leaky_relu(conv2->forward(x), 0.2);
  }

  nn::Conv2d conv0;
  nn::Conv2d conv1;
  nn::Conv2d conv2;
  // n.Conv2d(C_in, C_out, (3, 3), padding=1),
  // nn.LeakyReLU(negative_slope=0.2),
  // nn.Conv2d(C_out, C_out, (3, 3), padding=1),
  // nn.LeakyReLU(negative_slope=0.2),
  // nn.Conv2d(C_out, C_out, (3, 3), padding=1),
  // nn.LeakyReLU(negative_slope=0.2),
};
TORCH_MODULE(DB);

// nn::Sequential Encoder (
//   EB0(3, 48),
//   nn::AvgPool2d(nn::AvgPool2dOptions(2).stride(2).ceil_mode(true).count_include_pad(false)),
//   EB1(48, 80),
//   nn::AvgPool2d(nn::AvgPool2dOptions(2).stride(2).ceil_mode(true).count_include_pad(false)),
//   EB2(80, 144),
//   nn::AvgPool2d(nn::AvgPool2dOptions(2).stride(2).ceil_mode(true).count_include_pad(false)),
//   EB3(155, 256),
//   nn::AvgPool2d(nn::AvgPool2dOptions(2).stride(2).ceil_mode(true).count_include_pad(false)),
//   EB4(256, 384)
// )
struct EncoderImpl : nn::Module
{
  EncoderImpl() : eb0(3, 48),
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

  std::vector<torch::Tensor> forward(torch::Tensor x)
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

  EB eb0;
  EB eb1;
  EB eb2;
  EB eb3;
  EB eb4;
  nn::AvgPool2d avgpool;
  // self.EB0 = EB(3, 48)
  // # self.TD0 = nn.AvgPool2d((2, 2), (2, 2), ceil_mode=True, count_include_pad=False)
  // self.EB1 = EB(48, 80)
  // # self.TD1 = nn.AvgPool2d((2, 2), (2, 2), ceil_mode=True, count_include_pad=False)
  // self.EB2 = EB(80, 144)
  // # self.TD2 = nn.AvgPool2d((2, 2), (2, 2), ceil_mode=True, count_include_pad=False)
  // self.EB3 = EB(144, 256)
  // # self.TD3 = nn.AvgPool2d((2, 2), (2, 2), ceil_mode=True, count_include_pad=False)
  // self.EB4 = EB(256, 384)
};
TORCH_MODULE(Encoder);

struct DecoderImpl : nn::Module
{
  DecoderImpl() : db4(1920, 384),
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

  torch::Tensor forward(std::vector<torch::Tensor> &cc4)
  {
    torch::Tensor x;

    x = nn::functional::interpolate(db4->forward(cc4[0]), intOpts);
    x = torch::cat({x, cc4[1]}, 1);
    x = nn::functional::interpolate(db3->forward(x), intOpts);
    x = torch::cat({x, cc4[2]}, 1);
    x = nn::functional::interpolate(db2->forward(x), intOpts);
    x = torch::cat({x, cc4[3]}, 1);
    x = nn::functional::interpolate(db1->forward(x), intOpts);
    x = torch::cat({x, cc4[4]}, 1);
    return db0->forward(x);
  }

  DB db4;
  DB db3;
  DB db2;
  DB db1;
  DB db0;
  //std::vector<double> scale = {2, 2};
  nn::functional::InterpolateFuncOptions intOpts = nn::functional::InterpolateFuncOptions().scale_factor({2, 2});

  // self.DB4 = DB(1920, 384)
  // self.DB3 = DB(1664, 384)
  // self.DB2 = DB(1104, 240)
  // self.DB1 = DB(640, 136)
  // self.DB0 = DB(376, 80)
};
TORCH_MODULE(Decoder);

// class OutnX(nn.Module):
//     def __init__(self):
//         super(OutnX, self).__init__()
//         self.net = nn.Sequential(
//             nn.Conv2d(80, 40, (3, 3), padding=1),
//             nn.LeakyReLU(negative_slope=0.2),
//             nn.Conv2d(40, 3, (3, 3), padding=1)
//         )

//     def forward(self, decodes, inputFrame):
//         x = self.net(decodes)
//         return x + inputFrame

struct OutnXImpl : nn::Module
{
  OutnXImpl() : conv0(nn::Conv2dOptions(80, 40, 3).padding(1)),
                conv1(nn::Conv2dOptions(40, 3, 3).padding(1))
  {
    register_module("Conv3x3_0", conv0);
    register_module("Conv3x3_o", conv1);
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor inputFrame)
  {
    x = torch::leaky_relu(conv0->forward(x), 0.2);
    return conv1->forward(x) + inputFrame;
  }

  nn::Conv2d conv0;
  nn::Conv2d conv1;
};
TORCH_MODULE(OutnX);

// class UBnX(nn.Module):
//     def __init__(self):
//         super(UBnX, self).__init__()
//         self.net = nn.Sequential(
//             nn.Conv2d(80, 80, (3, 3), padding=1),
//             nn.LeakyReLU(negative_slope=0.2),
//             nn.Conv2d(80, 80, (3, 3), padding=1),
//             nn.LeakyReLU(negative_slope=0.2)
//         )

//     def forward(self, x):
//         return self.net(x)

struct UBnXImpl : nn::Module
{
  UBnXImpl() : conv0(nn::Conv2dOptions(80, 80, 3).padding(1)),
               conv1(nn::Conv2dOptions(80, 80, 3).padding(1))
  {
    register_module("Conv3x3_0", conv0);
    register_module("Conv3x3_1", conv1);
  }

  torch::Tensor forward(torch::Tensor x)
  {
    x = torch::leaky_relu(conv0->forward(x), 0.2);
    return torch::leaky_relu(conv1->forward(x), 0.2);
  }

  nn::Conv2d conv0;
  nn::Conv2d conv1;
};
TORCH_MODULE(UBnX);

// self.Encoder = Encoder()
// self.Decoder = Decoder()
// self.Out1X = OutnX()
// self.Out2X = OutnX()
// self.UB2X = UBnX()

// self.ec0 = None
// self.ec1 = None
// self.ec2 = None
// self.ec3 = None
// self.ec4 = None
// self.cFrame = None
// self.nFrame = None

// cc40 = torch.cat((self.ec0[0], self.ec1[0], self.ec2[0], self.ec3[0], self.ec4[0]), dim=1)
// cc41 = torch.cat((self.ec0[1], self.ec1[1], self.ec2[1], self.ec3[1], self.ec4[1]), dim=1)
// cc42 = torch.cat((self.ec0[2], self.ec1[2], self.ec2[2], self.ec3[2], self.ec4[2]), dim=1)
// cc43 = torch.cat((self.ec0[3], self.ec1[3], self.ec2[3], self.ec3[3], self.ec4[3]), dim=1)
// cc44 = torch.cat((self.ec0[4], self.ec1[4], self.ec2[4], self.ec3[4], self.ec4[4]), dim=1)
// decodes = self.Decoder(cc44, cc43, cc42, cc41, cc40)
// out1x = self.Out1X(decodes, self.cFrame)
// decodes2x = F.interpolate(decodes, scale_factor=2.)
// out1x2x = F.interpolate(out1x, scale_factor=2.)
// decodes2x = self.UB2X(decodes2x)
// out2x = self.Out2X(decodes2x, out1x2x)
// self.cFrame = self.nFrame
// self.nFrame = inputFrame

// SRModelImpl::SRModelImpl() : encoder(), decoder(), out1X(), out2X(), ub2X()
// {
//   register_module("Encoder", encoder);
//   register_module("Decoder", decoder);
//   register_module("Out1X", out1X);
//   register_module("Out2X", out2X);
//   register_module("UB2X", ub2X);
//   i = 0;
// }

// void SRModelImpl::load_frame(torch::Tensor inputFrame1, torch::Tensor inputFrame2)
// {
//   auto x = encoder->forward(inputFrame1);
//   // std::vector<torch::Tensor> x1(x);
//   // std::vector<torch::Tensor> x2(x);
//   // std::vector<torch::Tensor> x3(x);
//   ecs.push_back(x);
//   ecs.push_back(x);
//   ecs.push_back(x);
//   ecs.push_back(encoder->forward(inputFrame2));
//   ecs.push_back(x);
//   cFrame = inputFrame1.clone();
//   nFrame = inputFrame2.clone();
// }

// torch::Tensor SRModelImpl::forward(torch::Tensor x)
// {

//   ecs.at((4 + i) % 5) = encoder->forward(x);
//   std::vector<torch::Tensor> cc4;
//   for (int j = 4; j >= 0; j--)
//     cc4.emplace_back(torch::cat({ecs[(0 + i) % 5][j], ecs[(1 + i) % 5][j], ecs[(2 + i) % 5][j], ecs[(3 + i) % 5][j], ecs[(4 + i) % 5][j]}, 1));

//   auto decs = decoder->forward(cc4);
//   auto o1x = out1X->forward(decs, cFrame);
//   //auto dec2x = nn::functional::interpolate(decs, nn::functional::InterpolateFuncOptions().scale_factor(scale));
//   auto dec2x = nn::functional::interpolate(decs, intOpts);
//   auto o1x2x = nn::functional::interpolate(o1x, intOpts);
//   dec2x = ub2X->forward(dec2x);
//   auto o2x = out2X->forward(dec2x, o1x2x);
//   cFrame = nFrame;
//   nFrame = x;
//   return o2x;
// }

struct SRModelImpl : nn::Module
{
  SRModelImpl() : encoder(), decoder(), out1X(), out2X(), ub2X()
  {
    register_module("Encoder", encoder);
    register_module("Decoder", decoder);
    register_module("Out1X", out1X);
    register_module("Out2X", out2X);
    register_module("UB2X", ub2X);
    i = 0;
    //intOpts.scale_factor({2, 2});
  }

  void load_frame(torch::Tensor inputFrame1, torch::Tensor inputFrame2)
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
  }

  torch::Tensor forward(torch::Tensor x)
  {

    ecs.at((4+i) % 5) = encoder->forward(x);
    std::vector<torch::Tensor> cc4;
    for (int j =4; j >= 0; j--)
      cc4.emplace_back(torch::cat({ecs[(0+i) % 5][j], ecs[(1+i) % 5][j], ecs[(2+i) % 5][j],ecs[(3+i) % 5][j], ecs[(4+i) % 5][j]}, 1));

    auto decs = decoder->forward(cc4);
    auto o1x = out1X->forward(decs, cFrame);
    //auto dec2x = nn::functional::interpolate(decs, nn::functional::InterpolateFuncOptions().scale_factor(scale));
    auto dec2x = nn::functional::interpolate(decs, intOpts);
    auto o1x2x = nn::functional::interpolate(o1x, intOpts);
    dec2x = ub2X->forward(dec2x);
    auto o2x = out2X->forward(dec2x, o1x2x);
    cFrame = nFrame;
    nFrame = x;
    return o2x;
    // decodes = self.Decoder(cc44, cc43, cc42, cc41, cc40)
    // out1x = self.Out1X(decodes, self.cFrame)
    // decodes2x = F.interpolate(decodes, scale_factor=2.)
    // out1x2x = F.interpolate(out1x, scale_factor=2.)
    // decodes2x = self.UB2X(decodes2x)
    // out2x = self.Out2X(decodes2x, out1x2x)
    // self.cFrame = self.nFrame
    // self.nFrame = inputFrame
  }

  Encoder encoder;
  Decoder decoder;
  OutnX out1X;
  OutnX out2X;
  UBnX ub2X;
  std::vector<std::vector<torch::Tensor>> ecs;
  torch::Tensor cFrame;
  torch::Tensor nFrame;
  int i;
  //std::vector<double> scale = {2, 2};
  nn::functional::InterpolateFuncOptions intOpts = nn::functional::InterpolateFuncOptions().scale_factor({2,2});
};
TORCH_MODULE(SRModel);

int main(int argc, const char *argv[])
{
  torch::manual_seed(1);

  // Create the device we pass around based on whether CUDA is available.
  torch::Device device(torch::kCPU);
  // if (torch::cuda::is_available()) {
  //   std::cout << "CUDA is available! Training on GPU." << std::endl;
  //   device = torch::Device(torch::kCUDA);
  // }

  SRModel mod = SRModel();
  //mod.to(device);
  mod->to(device);
  //mod->state()
  mod->load_frame(torch::randn({1, 3, 640, 480}), torch::randn({1, 3, 640, 480}));
  //at::print(mod->forward(torch::randn({1, 3, 640, 480})));
  mod->forward(torch::randn({1, 3, 640, 480}));
  //torch::pickle_save() //(mod, "./test_modcpp.pt");
  PrintSateDict(*mod);
  //SaveStateDict(*mod, "state_dict.weigts");
  //LoadStateDictJson(*mod, "../../tlv2_std.json");
  LoadStateDict(*mod, "../../tlv2_std.dat");
  torch::save(mod, "./tlv2_std_cpp.pt");
  std::cout << "Down" << std::endl;
  return 0;
}