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

#include <stdio.h>
#include <math.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <iostream>
#include <rvfloats.h>
#include <util.h>
#include "svdpi.h"
// #include "verilated_vpi.h"
#include "VX_config.h"

#include <bit>
#include "half.h"

extern "C" {
  void dpi_fadd(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fsub(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fmul(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fmadd(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fmsub(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fnmadd(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fnmsub(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);

  void dpi_fdiv(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_fsqrt(bool enable, int dst_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);

  void dpi_ftoi(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_ftou(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_itof(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_utof(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags);
  void dpi_f2f(bool enable, int dst_fmt, int64_t a, int64_t* result);
  
  void dpi_fclss(bool enable, int dst_fmt, int64_t a, int64_t* result);
  void dpi_fsgnj(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result);
  void dpi_fsgnjn(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result);
  void dpi_fsgnjx(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result);

  void dpi_flt(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
  void dpi_fle(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
  void dpi_feq(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
  void dpi_fmin(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);
  void dpi_fmax(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags);

  void dpi_hmma(bool enable, const svBitVecVal* A_tile, const svBitVecVal* B_tile, const svBitVecVal* C_tile, svBitVecVal* D_tile);
  void dpi_print_results(int wid, int octet, const svBitVecVal* A_tile, const svBitVecVal* B_tile, const svBitVecVal* C_tile, const svBitVecVal* D_tile);
}

inline uint64_t nan_box(uint32_t value) {
#ifdef FPU_RV64F
  return value | 0xffffffff00000000;
#else 
  return value;
#endif
}

inline bool is_nan_boxed(uint64_t value) {
#ifdef FPU_RV64F
  return (uint32_t(value >> 32) == 0xffffffff);
#else
  __unused (value);
  return true;
#endif
}

inline int64_t check_boxing(int64_t a) {  
  if (!is_nan_boxed(a)) {
    return nan_box(0x7fc00000); // NaN
  }
  return a;
}

void dpi_fadd(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) {
    *result = rv_fadd_d(a, b, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fadd_s(check_boxing(a), check_boxing(b), (*frm & 0x7), fflags));
  }
}

void dpi_fsub(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) {
    *result = rv_fsub_d(a, b, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fsub_s(check_boxing(a), check_boxing(b), (*frm & 0x7), fflags));
  }
}

void dpi_fmul(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fmul_d(a, b, (*frm & 0x7), fflags); 
  } else {
    *result = nan_box(rv_fmul_s(check_boxing(a), check_boxing(b), (*frm & 0x7), fflags));
  }
}

void dpi_fmadd(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fmadd_d(a, b, c, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fmadd_s(check_boxing(a), check_boxing(b), check_boxing(c), (*frm & 0x7), fflags));
  }
}

void dpi_fmsub(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fmsub_d(a, b, c, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fmsub_s(check_boxing(a), check_boxing(b), check_boxing(c), (*frm & 0x7), fflags));
  }
}

void dpi_fnmadd(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fnmadd_d(a, b, c, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fnmadd_s(check_boxing(a), check_boxing(b), check_boxing(c), (*frm & 0x7), fflags));
  }
}

void dpi_fnmsub(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t c, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fnmsub_d(a, b, c, (*frm & 0x7), fflags);
  } else {
    *result = nan_box(rv_fnmsub_s(check_boxing(a), check_boxing(b), check_boxing(c), (*frm & 0x7), fflags));
  }
}

void dpi_fdiv(bool enable, int dst_fmt, int64_t a, int64_t b, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fdiv_d(a, b, (*frm & 0x7), fflags); 
  } else {
    *result = nan_box(rv_fdiv_s(check_boxing(a), check_boxing(b), (*frm & 0x7), fflags));
  }
}

void dpi_fsqrt(bool enable, int dst_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fsqrt_d(a, (*frm & 0x7), fflags); 
  } else {
    *result = nan_box(rv_fsqrt_s(check_boxing(a), (*frm & 0x7), fflags));
  }
}

void dpi_ftoi(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) {
    if (src_fmt) { 
      *result = rv_ftol_d(a, (*frm & 0x7), fflags);
    } else {
      *result = rv_ftol_s(check_boxing(a), (*frm & 0x7), fflags);
    }
  } else {    
    if (src_fmt) { 
      *result = sext<uint64_t>(rv_ftoi_d(a, (*frm & 0x7), fflags), 32);
    } else {
      *result = sext<uint64_t>(rv_ftoi_s(check_boxing(a), (*frm & 0x7), fflags), 32);
    }
  }
}

void dpi_ftou(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) {
    if (src_fmt) { 
      *result = rv_ftolu_d(a, (*frm & 0x7), fflags);
    } else {
      *result = rv_ftolu_s(check_boxing(a), (*frm & 0x7), fflags);
    }
  } else {    
    if (src_fmt) { 
      *result = sext<uint64_t>(rv_ftou_d(a, (*frm & 0x7), fflags), 32);
    } else {
      *result = sext<uint64_t>(rv_ftou_s(check_boxing(a), (*frm & 0x7), fflags), 32); 
    }
  }
}

void dpi_itof(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) {
    if (src_fmt) { 
      *result = rv_ltof_d(a, (*frm & 0x7), fflags);
    } else { 
      *result = rv_itof_d(a, (*frm & 0x7), fflags);
    }
  } else {
    if (src_fmt) { 
      *result = nan_box(rv_ltof_s(a, (*frm & 0x7), fflags)); 
    } else { 
      *result = nan_box(rv_itof_s(a, (*frm & 0x7), fflags)); 
    }
  }
}

void dpi_utof(bool enable, int dst_fmt, int src_fmt, int64_t a, const svBitVecVal* frm, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) {
    if (src_fmt) { 
      *result = rv_lutof_d(a, (*frm & 0x7), fflags);
    } else { 
      *result = rv_utof_d(a, (*frm & 0x7), fflags);
    }
  } else {
    if (src_fmt) { 
      *result = nan_box(rv_lutof_s(a, (*frm & 0x7), fflags));
    } else { 
      *result = nan_box(rv_utof_s(a, (*frm & 0x7), fflags));
    }
  }
}

void dpi_f2f(bool enable, int dst_fmt, int64_t a, int64_t* result) {
  if (!enable) 
    return;
  if (dst_fmt) {
    *result = rv_ftod((int32_t)check_boxing(a));
  } else {
    *result = nan_box(rv_dtof(a));
  }
}

void dpi_fclss(bool enable, int dst_fmt, int64_t a, int64_t* result) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fclss_d(a); 
  } else { 
    *result = rv_fclss_s(check_boxing(a)); 
  }
}

void dpi_fsgnj(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fsgnj_d(a, b); 
  } else {
    *result = nan_box(rv_fsgnj_s(check_boxing(a), check_boxing(b)));
  }
}

void dpi_fsgnjn(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fsgnjn_d(a, b); 
  } else {
    *result = nan_box(rv_fsgnjn_s(check_boxing(a), check_boxing(b)));
  }
}

void dpi_fsgnjx(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fsgnjx_d(a, b); 
  } else {
    *result = nan_box(rv_fsgnjx_s(check_boxing(a), check_boxing(b)));
  }
}

void dpi_flt(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) {
    *result = rv_flt_d(a, b, fflags); 
  } else {
    *result = rv_flt_s(check_boxing(a), check_boxing(b), fflags);
  }
}

void dpi_fle(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fle_d(a, b, fflags); 
  } else {
    *result = rv_fle_s(check_boxing(a), check_boxing(b), fflags);
  }
}

void dpi_feq(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_feq_d(a, b, fflags); 
  } else {
    *result = rv_feq_s(check_boxing(a), check_boxing(b), fflags);
  }
}

void dpi_fmin(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fmin_d(a, b, fflags); 
  } else {
    *result = nan_box(rv_fmin_s(check_boxing(a), check_boxing(b), fflags));
  }
}

void dpi_fmax(bool enable, int dst_fmt, int64_t a, int64_t b, int64_t* result, svBitVecVal* fflags) {
  if (!enable) 
    return;
  if (dst_fmt) { 
    *result = rv_fmax_d(a, b, fflags); 
  } else {
    *result = nan_box(rv_fmax_s(check_boxing(a), check_boxing(b), fflags));
  }
}

// A is M * K, B is K * M, C is M * M, D is M * M
#define M 4
#define K 2 // FIXME: 4x4x1 / cycle / octet!

// all row major
float c_A_tile[M][K];
float c_B_tile[K][M];
float c_C_tile[M][M];
float c_D_tile[M][M];

// code assumes that svBitVecVal is basically a uint32_t
static_assert(sizeof(svBitVecVal) == 4);

void clear_float_array(float* c_tile, int rows, int cols) {
  for (int i = 0; i < rows; i += 1) {
    for (int j = 0; j < cols; j += 1) {
      int index = i * cols + j;
      c_tile[index] = 0.0f;
    }
  }
}

void fill_float_array(const svBitVecVal* sv_tile, float* c_tile, int rows, int cols) {
  
  for (int i = 0; i < rows; i += 1) {
    for (int j = 0; j < cols; j += 1) {
      int index = i * cols + j;
      svBitVecVal sv_val = sv_tile[index];

      uint32_t c_val = sv_val;
      float c_float;

      memcpy(&c_float, &c_val, sizeof(c_float));
      c_tile[index] = c_float;
      
      // std::cout << c_float << " ";
    }
    // std::cout << std::endl;
  }
}

void write_float_array(svBitVecVal* sv_tile, float* c_tile, int rows, int cols) {
  for (int i = 0; i < rows; i += 1) {
    for (int j = 0; j < cols; j += 1) {
      int index = i * cols + j;
      svBitVecVal* sv_val = &sv_tile[index];

      float c_float = c_tile[index];
      memcpy(sv_val, &c_float, sizeof(c_float));

      // std::cout << c_float << " ";
    }
    // std::cout << std::endl;
  }
}

void dpi_hmma(bool enable, const svBitVecVal* A_tile, const svBitVecVal* B_tile, const svBitVecVal* C_tile, svBitVecVal* D_tile) {
  if (!enable) {
    return;
  }
  clear_float_array(&c_A_tile[0][0], M, K);
  clear_float_array(&c_B_tile[0][0], K, M);
  clear_float_array(&c_C_tile[0][0], M, M);
  clear_float_array(&c_D_tile[0][0], M, M);

  // std::cout << "A: " << std::endl;
  fill_float_array(A_tile, &c_A_tile[0][0], M, K);
  // std::cout << "B: " << std::endl;
  fill_float_array(B_tile, &c_B_tile[0][0], K, M);
  // std::cout << "C: " << std::endl;
  fill_float_array(C_tile, &c_C_tile[0][0], M, M);
  
  for (int i = 0; i < M; i += 1) {
    for (int j = 0; j < M; j += 1) {
      float accum = c_C_tile[i][j];
      for (int k = 0; k < K; k += 1) {
        accum += c_A_tile[i][k] * c_B_tile[k][j];
      }
      c_D_tile[i][j] = accum;
    }
  }

  write_float_array(D_tile, &c_D_tile[0][0], M, M);
}

// 1 copy per warp
float A_tile_full[4][16][8];
float B_tile_full[4][8][16];
float C_tile_full[4][16][16];
float D_tile_full[4][16][16];
int steps[4];

void print_array(float* array, int rows, int cols) {
  for (int i = 0; i < rows; i += 1) {
    for (int j = 0; j < cols; j += 1) {
      std::cout << array[i*cols+j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::endl;
}

void dpi_print_results(int wid, int octet, const svBitVecVal* A_tile, const svBitVecVal* B_tile, const svBitVecVal* C_tile, const svBitVecVal* D_tile) {
  // std::cout << "A: " << std::endl;
  fill_float_array(A_tile, &c_A_tile[0][0], M, K);
  // std::cout << "B: " << std::endl;
  fill_float_array(B_tile, &c_B_tile[0][0], K, M);
  // std::cout << "C: " << std::endl;
  fill_float_array(C_tile, &c_C_tile[0][0], M, M);
  // for some reason this still holds onto old value? very strange
  // std::cout << "D: " << std::endl;
  fill_float_array(D_tile, &c_D_tile[0][0], M, M);

  int octet_row_offset;
  int octet_col_offset;
  switch(octet) {
  case 0:
    octet_row_offset = 0;
    octet_col_offset = 0;
    break;
  case 1:
    octet_row_offset = 8;
    octet_col_offset = 0;
    break;
  case 2:
    octet_row_offset = 0;
    octet_col_offset = 8;
    break;
  case 3:
    octet_row_offset = 8;
    octet_col_offset = 8;
    break;
  }

  int step_row_offset;
  int step_col_offset;
  int step = (steps[wid] % 16) / 4;
  int set = (steps[wid] / 16);
  switch(step) {
  case 0:
    step_row_offset = 0;
    step_col_offset = 0;
    break;
  case 1:
    step_row_offset = 2;
    step_col_offset = 0;
    break;
  case 2:
    step_row_offset = 0;
    step_col_offset = 4;
    break;
  case 3:
    step_row_offset = 2;
    step_col_offset = 4;
    break;
  }
  
  if (steps[0] >= 48) {
    // std::cout << "octet " << octet << " step " << steps[0] << "\n";
    // print_array(&c_D_tile[0][0], 4, 4); 
  }

  D_tile_full[wid][octet_row_offset+step_row_offset+0][octet_col_offset+step_col_offset+0] = c_D_tile[0][0];
  D_tile_full[wid][octet_row_offset+step_row_offset+0][octet_col_offset+step_col_offset+1] = c_D_tile[0][1];
  D_tile_full[wid][octet_row_offset+step_row_offset+0][octet_col_offset+step_col_offset+2] = c_D_tile[0][2];
  D_tile_full[wid][octet_row_offset+step_row_offset+0][octet_col_offset+step_col_offset+3] = c_D_tile[0][3];
  D_tile_full[wid][octet_row_offset+step_row_offset+1][octet_col_offset+step_col_offset+0] = c_D_tile[1][0];
  D_tile_full[wid][octet_row_offset+step_row_offset+1][octet_col_offset+step_col_offset+1] = c_D_tile[1][1];
  D_tile_full[wid][octet_row_offset+step_row_offset+1][octet_col_offset+step_col_offset+2] = c_D_tile[1][2];
  D_tile_full[wid][octet_row_offset+step_row_offset+1][octet_col_offset+step_col_offset+3] = c_D_tile[1][3];
  D_tile_full[wid][octet_row_offset+step_row_offset+4][octet_col_offset+step_col_offset+0] = c_D_tile[2][0];
  D_tile_full[wid][octet_row_offset+step_row_offset+4][octet_col_offset+step_col_offset+1] = c_D_tile[2][1];
  D_tile_full[wid][octet_row_offset+step_row_offset+4][octet_col_offset+step_col_offset+2] = c_D_tile[2][2];
  D_tile_full[wid][octet_row_offset+step_row_offset+4][octet_col_offset+step_col_offset+3] = c_D_tile[2][3];
  D_tile_full[wid][octet_row_offset+step_row_offset+5][octet_col_offset+step_col_offset+0] = c_D_tile[3][0];
  D_tile_full[wid][octet_row_offset+step_row_offset+5][octet_col_offset+step_col_offset+1] = c_D_tile[3][1];
  D_tile_full[wid][octet_row_offset+step_row_offset+5][octet_col_offset+step_col_offset+2] = c_D_tile[3][2];
  D_tile_full[wid][octet_row_offset+step_row_offset+5][octet_col_offset+step_col_offset+3] = c_D_tile[3][3];

  if (octet == 0 || octet == 1) {
    octet_row_offset = octet * 8;
    if (step == 0) {
      step_row_offset = 0; 
    }
    if (step == 1) {
      step_row_offset = 2;
    }
    if (step == 0 || step == 1) {
      A_tile_full[wid][octet_row_offset+step_row_offset+0][set*2+0] = c_A_tile[0][0];
      A_tile_full[wid][octet_row_offset+step_row_offset+0][set*2+1] = c_A_tile[0][1];
      A_tile_full[wid][octet_row_offset+step_row_offset+1][set*2+0] = c_A_tile[1][0];
      A_tile_full[wid][octet_row_offset+step_row_offset+1][set*2+1] = c_A_tile[1][1];
      A_tile_full[wid][octet_row_offset+step_row_offset+4][set*2+0] = c_A_tile[2][0];
      A_tile_full[wid][octet_row_offset+step_row_offset+4][set*2+1] = c_A_tile[2][1];
      A_tile_full[wid][octet_row_offset+step_row_offset+5][set*2+0] = c_A_tile[3][0];
      A_tile_full[wid][octet_row_offset+step_row_offset+5][set*2+1] = c_A_tile[3][1];
    }
  }

  if (octet == 0 || octet == 2) {
    octet_col_offset = octet * 4;
    if (step == 0) {
      step_col_offset = 0; 
    }
    else if (step == 2) {
      step_col_offset = 4;
    }
    if (step == 0 || step == 2) {
      B_tile_full[wid][set*2+0][octet_col_offset+step_col_offset+0] = c_B_tile[0][0];
      B_tile_full[wid][set*2+0][octet_col_offset+step_col_offset+1] = c_B_tile[0][1];
      B_tile_full[wid][set*2+0][octet_col_offset+step_col_offset+2] = c_B_tile[0][2];
      B_tile_full[wid][set*2+0][octet_col_offset+step_col_offset+3] = c_B_tile[0][3];
      B_tile_full[wid][set*2+1][octet_col_offset+step_col_offset+0] = c_B_tile[1][0];
      B_tile_full[wid][set*2+1][octet_col_offset+step_col_offset+1] = c_B_tile[1][1];
      B_tile_full[wid][set*2+1][octet_col_offset+step_col_offset+2] = c_B_tile[1][2];
      B_tile_full[wid][set*2+1][octet_col_offset+step_col_offset+3] = c_B_tile[1][3];
    }
  }

  steps[wid] += 1;
  if (steps[wid] % 32 == 0) {
    steps[wid] = 0;
    std::cout << "warp " << wid << " finished wmma\n";
    std::cout << "A tile" << "\n";
    print_array(&A_tile_full[wid][0][0], 16, 8);
    std::cout << "B tile" << "\n";
    print_array(&B_tile_full[wid][0][0], 8, 16);
    // std::cout << "C tile" << "\n";
    // print_array(&C_tile_full[wid][0][0], 16, 16);
    std::cout << "D tile" << "\n";
    print_array(&D_tile_full[wid][0][0], 16, 16);
  }
}
