// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "vl_simulator.h"
#include "VVX_tensor_tb.h"
#include <iostream>

#include <half.hpp>

#define MAX_TICKS 20

#ifndef TRACE_START_TIME
#define TRACE_START_TIME 0ull
#endif

#ifndef TRACE_STOP_TIME
#define TRACE_STOP_TIME -1ull
#endif

#define CHECK(x)                                  \
   do {                                           \
     if (x)                                       \
       break;                                     \
     std::cout << "FAILED: " << #x << std::endl;  \
	   std::abort();			                          \
   } while (false)

static uint64_t timestamp = 0;
static bool trace_enabled = false;
static uint64_t trace_start_time = TRACE_START_TIME;
static uint64_t trace_stop_time  = TRACE_STOP_TIME;

double sc_time_stamp() { 
  return timestamp;
}

bool sim_trace_enabled() {
  if (timestamp >= trace_start_time 
   && timestamp < trace_stop_time)
    return true;
  return trace_enabled;
}

void sim_trace_enable(bool enable) {
  trace_enabled = enable;
}

using Device = VVX_tensor_tb;
using half_float::half;

static_assert(sizeof(half) == 2);
uint32_t half2bits(half h) {
    uint16_t half_bits;
    memcpy(&half_bits, &h, sizeof(half));
    return half_bits;
}

uint32_t float2bits(float f) {
  uint32_t float_bits;
  memcpy(&float_bits, &f, sizeof(f));
  return float_bits;
}

float bits2float(uint32_t b) {
  float f;
  memcpy(&f, &b, sizeof(b));
  return f;
}

// A is M * K, B is K * K * M, C is M * M, D is M * M
#define M 4
#define K 2

// row, column
float A_tile[M][K];
float B_tile[K][M];
float C_tile[M][M];
float D_tile[M][M];

void initialize_test_data() {
  for (int i = 0; i < M; i += 1) {
    for (int j = 0; j < K; j += 1) {
      A_tile[i][j] = (float) (i * K + j);
    }
  }

  for (int i = 0; i < K; i += 1) {
    for (int j = 0; j < M; j += 1) {
      B_tile[i][j] = (float) (j * K + i);
    }
  }
  for (int i = 0; i < M; i += 1) {
    for (int j = 0; j < M; j += 1) {
      C_tile[i][j] = (float) (i * j);
    }
  }
}

void write_test_data(vl_simulator<Device>& sim) {
  for (int i = 0; i < M; i += 1) {
    for (int j = 0; j < K; j += 1) {
      int index = (i * K + j);
      uint32_t A_bits = float2bits(A_tile[i][j]);
      sim->A_tile[index] = A_bits;
    }
  }

  for (int i = 0; i < K; i += 1) {
    for (int j = 0; j < M; j += 1) {
      int index = (i * M + j);
      uint32_t B_bits = float2bits(B_tile[i][j]);
      sim->B_tile[index] = B_bits;
    }
  }

  for (int i = 0; i < M; i += 1) {
    for (int j = 0; j < M; j += 1) {
      int index = (i * M + j);
      uint32_t C_bits = float2bits(C_tile[i][j]);
      sim->C_tile[index] = C_bits;
    }
  }
}

void read_result(vl_simulator<Device>& sim) {
  for (int i = 0; i < M; i += 1) {
    for (int j = 0; j < M; j += 1) {
      int index = (i * M + j);

      uint32_t D_bits = sim->D_tile[index];
      float f = bits2float(D_bits);
      D_tile[i][j] = f;
      std::cout << f << " ";
    }
    std::cout << std::endl;
  }
}

void expected() {
  for (int i = 0; i < M; i += 1) {
    for (int j = 0; j < M; j += 1) {
      float accum = C_tile[i][j];
      for (int k = 0; k < K; k += 1) {
        accum += A_tile[i][k] * B_tile[k][j];
      }

      std::cout << accum << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char **argv) {
  // Initialize Verilators variables
  Verilated::commandArgs(argc, argv);

  vl_simulator<Device> sim;

  initialize_test_data();
  // run test
  timestamp = sim.reset(0);


  // advance clock
  timestamp = sim.step(timestamp, 10);
  sim->valid_in = 1;
  write_test_data(sim);
  timestamp = sim.step(timestamp, 2);
  CHECK(sim->valid_out == 0);
  sim->valid_in = 0;
  timestamp = sim.step(timestamp, 2);
  CHECK(sim->valid_out == 0);
  timestamp = sim.step(timestamp, 2);
  CHECK(sim->valid_out == 0);
  timestamp = sim.step(timestamp, 2);
  CHECK(sim->valid_out == 1);
  read_result(sim);
  timestamp = sim.step(timestamp, 2);
  CHECK(sim->valid_out == 0);

  expected();

  std::cout << "PASSED!" << std::endl;
  std::cout << "Simulation time: " << std::dec << timestamp/2 << " cycles" << std::endl;

  return 0;
}