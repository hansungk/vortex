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

`include "VX_define.vh"

interface VX_tc_rf_if import VX_gpu_pkg::*; ();

    typedef struct packed {
        logic [ISSUE_WIS_W-1:0] wis;
        logic [`NR_BITS-1:0]    rs;
    } req_data_t;

    typedef struct packed {
        logic [`NUM_THREADS-1:0][`XLEN-1:0] data;
    } rsp_data_t;

    logic  req_valid;
    req_data_t req_data;

    rsp_data_t rsp_data;

    modport master (
        output req_valid,
        output req_data,

        input  rsp_data
    );

    modport slave (
        input  req_valid,
        input  req_data,

        output rsp_data
    );

endinterface
