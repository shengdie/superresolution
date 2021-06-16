#include "model_cppall.hpp"
#include "cxxopts.hpp"
//#include "state_loader.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstdlib>
#include <csignal>

cv::VideoWriter writer;

void mat2tensor(cv::Mat &input, at::Tensor &t)
{
  //auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
  //t = torch::from_blob(input.ptr<float>(), /*sizes=*/{1, input.rows, input.cols, 3}, options);
  //float* tensorData = frameTensor.data_ptr<float>();
  cv::Mat outputImg(input.rows, input.cols, CV_32FC3, t.data_ptr<float>());
  input.convertTo(outputImg, CV_32FC3, 1.0f / 255.0f);
  //outputImg = outputImg/255.0f;
  //t.permute({0, 3, 1, 2});
}

void tensor2mat(at::Tensor &t, cv::Mat &image)
{
  //t.permute({0, 2, 3, 1});
  image = cv::Mat(t.size(1), t.size(2), CV_8UC3, t.data_ptr<u_int8_t>());
  //image *= 255.;
  //image.convertTo(image, CV_8UC3);
}

void sigint_handler(int s){
    printf("Got signal %d, closing\n", s);
    writer.release();
    exit(1); 
}

cxxopts::ParseResult parse(int argc, char *argv[])
{
  try
  {
    //int argc_b = argc;
    cxxopts::Options options(argv[0], " - Super resolution of videos.");
    options
        .positional_help("[optional args]")
        .show_positional_help();

    options
        .add_options()("C,use_cpu", "Use cpu to run", cxxopts::value<bool>()->default_value("false"))
        ("c,codec", "Codec for output video", cxxopts::value<std::string>()->default_value("X264"))
        ("m,model", "Model to run super r", cxxopts::value<std::string>()->default_value("std_2x"))
        ("x,upsample", "upsample times, now only 2x supported", cxxopts::value<int>()->default_value("2"))
        ("i,input", "Input", cxxopts::value<std::string>())
        ("o,output", "Output file", cxxopts::value<std::string>())
        ("std", "Using std model", cxxopts::value<bool>()->default_value("false"))
        ("cg", "Using cg model", cxxopts::value<bool>()->default_value("false"))
        //   ("positional",
        //     "Positional arguments: these are the arguments that are entered "
        //     "without an option", cxxopts::value<std::vector<std::string>>())
        ("help", "Print help");

    options.parse_positional({"input", "output"});

    
    if (argc == 1) {
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }
    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("help"))
    {
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }
    

    return result;
  }
  catch (const cxxopts::OptionException &e)
  {
    std::cout << "error parsing options: " << e.what() << std::endl;
    exit(1);
  }
}

int main(int argc, char *argv[])
{
  struct sigaction sigIntHandler;

  sigIntHandler.sa_handler = sigint_handler;
  sigemptyset(&sigIntHandler.sa_mask);
  /* handle ctral + c  */
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);
  
  auto result = parse(argc, argv);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end; // = std::chrono::steady_clock::now();
  std::string codecstring = result["codec"].as<std::string>();
  if (codecstring.length() != 4)
  {
    std::cerr << "Error: the codec must be a 4 chars" << std::endl;
    exit(-1);
  }
  int codec = cv::VideoWriter::fourcc(codecstring[0], codecstring[1], codecstring[2], codecstring[3]);
  //torch::manual_seed(1);
  std::string filename = result["input"].as<std::string>();
  cv::VideoCapture capture(filename);
  cv::Mat frame;
  cv::Mat outframe;
  if (!capture.isOpened())
  {
    std::cout << "Error when reading video\n";
    return -1;
  }

  int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  int depth = 3;
  int frameCount = 0; //capture.get(cv::CAP_PROP_FRAME_COUNT);
  // bool grabed = capture.read(frame);
  // while (grabed)
  // {
  //   frameCount++;
  //   grabed = capture.read(frame);
  // }
  while (capture.grab())
  {
    frameCount++;
  }
  capture.release();
  capture = cv::VideoCapture(filename);
  std::cout << "Source Video: " << filename << std::endl;
  std::cout << "Height: " << height << ", Width: " << width << "." << std::endl;
  std::cout << "Frame count: " << frameCount << std::endl << std::endl;

  // Create the device we pass around based on whether CUDA is available.
  
  // } else {
  //   device = torch::Device(torch::kCPU);
  // }
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available())
  {
    std::cout << "CUDA is available! Running on GPU." << std::endl;
    device = torch::Device(torch::kCUDA);
  }
  torch::NoGradGuard no_grad;
  SRModel mod = SRModel();
  std::string modelFlag;
  std::cout << "==================================" << std::endl; 
  std::string model_file;
  if(result["std"].as<bool>()) {
    model_file = "/usr/local/share/supersr/supersr_std_2x.pt";
    std::cout << "Using model: supersr_std_2x" << std::endl;
  } else if(result["cg"].as<bool>()) {
    model_file = "/usr/local/share/supersr/supersr_cg_2x.pt";
    std::cout << "Using model: supersr_cg_2x" << std::endl;
  } else {
    model_file = result["model"].as<std::string>();
    if (model_file == "std_2x") {
      model_file = "/usr/local/share/supersr/supersr_std_2x.pt";
      std::cout << "Using model: supersr_std_2x" << std::endl;
    } else {
      std::cout << "Using model: " << model_file << std::endl;
    }
  }
  
  std::cout << "==================================" << std::endl; 
  if (model_file.find("std") != std::string::npos) {
    modelFlag = "_std_";
  } else if (model_file.find("cg") != std::string::npos) {
    modelFlag = "_cg_";
  } else {
    modelFlag = "";
  }
  
  torch::load(mod, model_file);
  //mod.to(device);
  mod->to(device);

  //torch::NoGradGuard no_grad;
  auto options = torch::TensorOptions().dtype(torch::kFloat32); //.device(torch::kCUDA, 0);
  auto frameTensor1 = torch::empty({1, height, width, depth}, options);
  auto frameTensor2 = torch::empty({1, height, width, depth}, options);
  auto frameTensor3 = torch::empty({1, height, width, depth}, options);
  capture >> frame;
  cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
  mat2tensor(frame, frameTensor1);
  capture >> frame;
  cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
  mat2tensor(frame, frameTensor2);
  frameCount -= 2;

  mod->load_frame(frameTensor1.permute({0, 3, 1, 2}).to(device), frameTensor2.permute({0, 3, 1, 2}).to(device));

  //cv::VideoWriter writer;


  //int codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
  double fps = capture.get(cv::CAP_PROP_FPS);
  std::string outputfile;
  if (!result.count("output")) {
    std::string::size_type pAt = filename.find_last_of('/') + 1;
    //if (pAt == -1) pAt = 0;
    std::string::size_type pBt = filename.find_last_of('.');
    //std::string::size_type lenn = pAt == -1 ? pBt : pBt - pAt;
    outputfile = filename.substr(pAt, pBt - pAt) + modelFlag + codecstring + "_2X.mp4";
  } else {
    outputfile = result["output"].as<std::string>();
  }
  std::cout << "Output Video: " << outputfile << std::endl << std::endl;

  writer.open(outputfile, codec, fps, frame.size() * 2);
  // check if we succeeded
  if (!writer.isOpened())
  {
    std::cerr << "Could not open the output video file for write\n";
    exit(-1);
  }
  //std::cout << "qulity: " << writer.get(cv::VIDEOWRITER_PROP_QUALITY) << std::endl;

  //while(!frame.empty())
  torch::Tensor outTensor;
  cv::Mat outframe1;
  int processed = -1;
  int minCount = 0;
  int timeleft = 0;
  int percent = 0;
  for (int j = 0; j < frameCount; j++)
  {
    capture >> frame;
    // if (frame.empty()) {
    //     std::cout << "Converting Finished\n";
    //     //writer.release();
    //     std::cout << "qulity: " << writer.get(cv::VIDEOWRITER_PROP_QUALITY) << std::endl;
    //     break;
    //     //return -1;
    // }
    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

    //torch::NoGradGuard no_grad;
    mat2tensor(frame, frameTensor3);
    //std::vector<torch::jit::IValue> inputs;
    //inputs.emplace_back(frameTensor3.permute({0, 3, 1, 2}).to(device));
    //cv::imwrite("test.png", )
    outTensor = mod->forward(frameTensor3.permute({0, 3, 1, 2}).to(device));
    outframe = cv::Mat(outTensor.size(2), outTensor.size(3), CV_8UC3, outTensor.permute({0, 2, 3, 1}).mul_(255.0f).clamp_(0, 255).to(torch::kU8).to(torch::kCPU).data_ptr<u_int8_t>());
    //at::print(module.forward(inputs).toTensor());
    //v::imwrite("test.png", outframe);
    cv::cvtColor(outframe, outframe1, cv::COLOR_RGB2BGR);
    writer.write(outframe1);
    if (j > processed)
    {
      //std::cout.flush();
      end = std::chrono::steady_clock::now();
      minCount = std::chrono::duration_cast<std::chrono::minutes>(end - begin).count();
      percent = (100 * (j + 1)) / (frameCount + 2);
      timeleft = minCount >= 2 ? (int)(std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() * 5.0f/3.0f / percent) : 0;
      std::cout << "\r"
                << "[" << std::string(percent / 5, '*') << std::string(100 / 5 - percent / 5, ' ') << "]";
      std::cout << minCount / 60 << "h:" << minCount % 60 << "min"
                << " [Frames " << j + 1 << "/" << frameCount + 2 << "][" << percent << "%][" << "Time left: " << timeleft / 60
                << "h:" << timeleft % 60 << "min]";
      //std::cout.flush();
      processed += 100;
      std::cout.flush();
    }
    //break;
  }
  outTensor = mod->forward(frameTensor3.permute({0, 3, 1, 2}).to(device));
  outframe = cv::Mat(outTensor.size(2), outTensor.size(3), CV_8UC3, outTensor.permute({0, 2, 3, 1}).mul_(255.0f).clamp_(0, 255).to(torch::kU8).to(torch::kCPU).data_ptr<u_int8_t>());
  cv::cvtColor(outframe, outframe1, cv::COLOR_RGB2BGR);
  writer.write(outframe1);
  outTensor = mod->forward(frameTensor3.permute({0, 3, 1, 2}).to(device));
  outframe = cv::Mat(outTensor.size(2), outTensor.size(3), CV_8UC3, outTensor.permute({0, 2, 3, 1}).mul_(255.0f).clamp_(0, 255).to(torch::kU8).to(torch::kCPU).data_ptr<u_int8_t>());
  cv::cvtColor(outframe, outframe1, cv::COLOR_RGB2BGR);
  writer.write(outframe1);

  std::cout << "\nDown" << std::endl;
  end = std::chrono::steady_clock::now();
  std::cout << "Time cost= " << std::chrono::duration_cast<std::chrono::minutes>(end - begin).count() << "[min]" << std::endl;
  return 0;
}
