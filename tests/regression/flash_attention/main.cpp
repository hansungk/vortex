#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <vector>
#include <cassert>
#include "common.h"
#include "half.hpp"

using half_float::half;
using half_float::half_cast;

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

int run_test(const kernel_arg_t& kernel_arg,
             uint32_t buf_size) {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.addr_o, buf_size));

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

  uint32_t dim_seqlen = 64;
  uint32_t dim_headdim = 64;

  using float_type = half;

  uint32_t dst_buf_size =
      dim_seqlen * dim_headdim * sizeof(ref_data[0]);

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  kernel_arg.addr_q = 0xa0000000;
  kernel_arg.addr_k = 0xa1000000;
  kernel_arg.addr_v = 0xa2000000;
  kernel_arg.addr_o = 0xc0000000;

  kernel_arg.dim_seqlen = dim_seqlen;
  kernel_arg.dim_headdim = dim_headdim;

  std::cout << "dev_addr_q=0x" << std::hex << kernel_arg.addr_q << std::endl;
  std::cout << "dev_addr_k=0x" << std::hex << kernel_arg.addr_k << std::endl;
  std::cout << "dev_addr_v=0x" << std::hex << kernel_arg.addr_v << std::endl;
  std::cout << "dev_addr_o=0x" << std::hex << kernel_arg.addr_o << std::endl;

  // allocate staging buffer
  {
    std::cout << "allocate staging buffer" << std::endl;
    uint32_t staging_buf_size = sizeof(kernel_arg_t);
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

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, dst_buf_size));
  std::cout << "PASSED!" << std::endl;

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  return 0;
}
