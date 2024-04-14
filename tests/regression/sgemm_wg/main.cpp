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

std::vector<float> src_a_data;
std::vector<float> src_b_data;
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

void generate_source_matrix(uint32_t dim_m, uint32_t dim_n, uint32_t dim_k) {
  src_a_data.resize(dim_m * dim_k);
  src_b_data.resize(dim_k * dim_n);

  for (uint32_t i = 0; i < src_a_data.size(); ++i) {
    src_a_data[i] = static_cast<float>(i);
    std::cout << "A: " << i << ": value=" << src_a_data[i] << std::endl;
  }
  for (uint32_t i = 0; i < src_b_data.size(); ++i) {
    src_b_data[i] = static_cast<float>(i);
    std::cout << "B: " << i << ": value=" << src_b_data[i] << std::endl;
  }
}

void generate_reference_matmul(uint32_t dim_m, uint32_t dim_n, uint32_t dim_k) {
  ref_data.resize(dim_m * dim_n);

  for (uint32_t i = 0; i < dim_m; ++i) {
    for (uint32_t j = 0; j < dim_n; ++j) {
      float ref = 0.0f;
      for (uint32_t k = 0; k < dim_k; ++k) {
        ref += src_a_data[dim_k * i + k] * src_b_data[dim_n * k + j];
      }
      ref_data.at(dim_n * i + j) = ref;
    }
  }
}

int run_test(const kernel_arg_t& kernel_arg,
             uint32_t buf_size,
             uint32_t dim_m, uint32_t dim_n) {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.addr_c, buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  {
    int errors = 0;
    auto buf_ptr = (float*)staging_buf.data();
    for (uint32_t i = 0; i < dim_m * dim_n; ++i) {
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

  // FIXME: hardcoded
  uint32_t dim_m = 32;
  uint32_t dim_n = 32;
  uint32_t dim_k = 32;

  generate_source_matrix(dim_m, dim_n, dim_k);
  generate_reference_matmul(dim_m, dim_n, dim_k);

  uint32_t src_a_buf_size = src_a_data.size() * sizeof(src_a_data[0]);
  uint32_t src_b_buf_size = src_b_data.size() * sizeof(src_b_data[0]);
  uint32_t dst_buf_size = ref_data.size() * sizeof(src_a_data[0]);

  std::cout << "buffer size: " << dst_buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  // RT_CHECK(vx_mem_alloc(device, src_a_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.addr_a));
  // RT_CHECK(vx_mem_alloc(device, src_b_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.addr_b));
  // RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.addr_c));
  kernel_arg.addr_a = 0x20000UL;
  kernel_arg.addr_b = 0x28000UL;
  kernel_arg.addr_c = 0xc0000000UL;

  kernel_arg.dim_m = dim_m;
  kernel_arg.dim_n = dim_n;
  kernel_arg.dim_k = dim_k;

  std::cout << "dev_addr_a=0x" << std::hex << kernel_arg.addr_a << std::endl;
  std::cout << "dev_addr_b=0x" << std::hex << kernel_arg.addr_b << std::endl;
  std::cout << "dev_addr_c=0x" << std::hex << kernel_arg.addr_c << std::endl;

  // allocate staging buffer
  {
    std::cout << "allocate staging buffer" << std::endl;
    uint32_t staging_buf_size = std::max<uint32_t>(
        src_a_buf_size,
        std::max<uint32_t>(
            src_b_buf_size,
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
        memcpy(buf_ptr, src_a_data.data(), src_a_data.size() * sizeof(float));
        RT_CHECK(vx_copy_to_dev(device, kernel_arg.addr_a, staging_buf.data(),
                                src_a_buf_size));

        std::cout << "uploading source A matrix to device, device mem address="
                  << std::hex << kernel_arg.addr_a << ", size=" << std::dec
                  << src_a_buf_size << " bytes\n";
        std::ofstream file("input.a.bin", std::ios::binary | std::ios::out);
        if (!file) {
        std::cerr << "error: failed to open args.bin for writing\n";
        exit(EXIT_FAILURE);
        }
        file.write(reinterpret_cast<char *>(buf_ptr), src_a_buf_size);
        file.close();
    }
    {
        auto buf_ptr = staging_buf.data();
        memcpy(buf_ptr, src_b_data.data(), src_b_data.size() * sizeof(float));
        RT_CHECK(vx_copy_to_dev(device, kernel_arg.addr_b, staging_buf.data(),
                                src_b_buf_size));

        std::cout << "uploading source B matrix to device, device mem address="
                  << std::hex << kernel_arg.addr_b << ", size=" << std::dec
                  << src_b_buf_size << " bytes\n";
        std::ofstream file("input.b.bin", std::ios::binary | std::ios::out);
        if (!file) {
        std::cerr << "error: failed to open args.bin for writing\n";
        exit(EXIT_FAILURE);
        }
        file.write(reinterpret_cast<char *>(buf_ptr), src_b_buf_size);
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
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.addr_c, staging_buf.data(), dst_buf_size));
  }

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, dst_buf_size, kernel_arg.dim_m, kernel_arg.dim_n));
  std::cout << "PASSED!" << std::endl;

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  return 0;
}
