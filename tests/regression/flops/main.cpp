#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <vector>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.bin";
uint32_t count = 0;

std::vector<float> src_data;
std::vector<float> ref_data;

vx_device_h device = nullptr;
std::vector<uint8_t> staging_buf;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h?")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    // vx_mem_free(device, kernel_arg.addr_a);
    // vx_mem_free(device, kernel_arg.addr_b);
    // vx_mem_free(device, kernel_arg.addr_c);
    vx_dev_close(device);
  }
}

void generate_source_data(size_t size) {
  src_data.resize(size);

  for (uint32_t i = 0; i < src_data.size(); ++i) {
    src_data[i] = static_cast<float>(i);
  }
}

void generate_reference_data(size_t size) {
  ref_data.resize(size);

  for (uint32_t i = 0; i < ref_data.size(); ++i) {
    ref_data[i] = static_cast<float>(i) * 1000.0f;
  }
}

int run_test(const kernel_arg_t& kernel_arg,
             uint32_t buf_size,
             uint32_t size) {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.addr_dst, buf_size));

  std::cout << "downloading result C matrix from device, device mem address="
            << std::hex << kernel_arg.addr_dst << ", size=" << std::dec
            << buf_size << " bytes\n";
  std::ofstream file("output.bin", std::ios::binary | std::ios::out);
  if (!file) {
    std::cerr << "error: failed to open output.bin for writing\n";
    exit(EXIT_FAILURE);
  }
  file.write(reinterpret_cast<char *>(staging_buf.data()), buf_size);
  file.close();

  std::ofstream ref_file("reference.bin", std::ios::binary | std::ios::out);
  if (!ref_file) {
    std::cerr << "error: failed to open reference.bin for writing\n";
    exit(EXIT_FAILURE);
  }
  ref_file.write(reinterpret_cast<char *>(ref_data.data()), buf_size);
  ref_file.close();

  // verify result
  std::cout << "verify result" << std::endl;
  {
    int errors = 0;
    auto buf_ptr = (float*)staging_buf.data();
    for (uint32_t i = 0; i < size; ++i) {
      float ref = ref_data.at(i);
      float cur = buf_ptr[i];
      if (std::abs((cur - ref) / ref) > 1e-6) {
        std::cout << "error at result #" << std::dec << i
                  << std::hex << ": actual=" << cur << ", expected=" << ref << std::endl;
        ++errors;
      }
    }
    if (errors != 0) {
      std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return 1;
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  size_t size = 64;

  generate_source_data(size);
  generate_reference_data(size);

  uint32_t src_buf_size = src_data.size() * sizeof(src_data[0]);
  uint32_t dst_buf_size = ref_data.size() * sizeof(ref_data[0]);

  std::cout << "buffer size: " << dst_buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  // RT_CHECK(vx_mem_alloc(device, src_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.addr_src));
  // RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.addr_dst));
  kernel_arg.addr_src = 0x20000UL;
  kernel_arg.addr_dst = 0xc0000000UL;
  kernel_arg.size = size;

  std::cout << "dev_addr_src=0x" << std::hex << kernel_arg.addr_src << std::endl;
  std::cout << "dev_addr_dst=0x" << std::hex << kernel_arg.addr_dst << std::endl;

  // allocate staging buffer
  {
    std::cout << "allocate staging buffer" << std::endl;
    uint32_t staging_buf_size = std::max<uint32_t>(
        src_buf_size,
        std::max<uint32_t>(
            src_buf_size,
            std::max<uint32_t>(dst_buf_size, sizeof(kernel_arg_t))));
    staging_buf.resize(staging_buf_size);
  }

  // upload kernel argument
  {
    std::cout << "upload kernel argument" << std::endl;
    auto buf_ptr = staging_buf.data();
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));

    std::cout << "uploading argument buffer to device, device mem address="
              << std::hex << KERNEL_ARG_DEV_MEM_ADDR << ", size=" << std::dec
              << sizeof(kernel_arg_t) << " bytes\n";
    std::ofstream file("args.bin", std::ios::binary | std::ios::out);
    if (!file) {
        std::cerr << "error: failed to open args.bin for writing\n";
        exit(EXIT_FAILURE);
    }
    file.write(reinterpret_cast<char *>(staging_buf.data()),
               sizeof(kernel_arg_t));
    file.close();
  }

  // upload source buffer
  {
    {
        auto buf_ptr = staging_buf.data();
        memcpy(buf_ptr, src_data.data(), src_data.size() * sizeof(float));
        RT_CHECK(vx_copy_to_dev(device, kernel_arg.addr_src, staging_buf.data(),
                                src_buf_size));

        std::cout << "uploading source data to device, device mem address="
                  << std::hex << kernel_arg.addr_src << ", size=" << std::dec
                  << src_buf_size << " bytes\n";
        std::ofstream file("input.a.bin", std::ios::binary | std::ios::out);
        if (!file) {
        std::cerr << "error: failed to open input.a.bin for writing\n";
        exit(EXIT_FAILURE);
        }
        file.write(reinterpret_cast<char *>(buf_ptr), src_buf_size);
        file.close();
    }
  }

  // clear destination buffer
  {
    std::cout << "clear destination buffer" << std::endl;
    auto buf_ptr = (int32_t*)staging_buf.data();
    for (uint32_t i = 0; i < ref_data.size(); ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.addr_dst, staging_buf.data(), dst_buf_size));
  }

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, dst_buf_size, kernel_arg.size));
  std::cout << "PASSED!" << std::endl;

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  return 0;
}
