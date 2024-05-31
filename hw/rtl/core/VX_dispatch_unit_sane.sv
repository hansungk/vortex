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

module VX_dispatch_unit_sane import VX_gpu_pkg::*; #(    
    parameter BLOCK_SIZE = 1,
    parameter NUM_LANES  = 1,
    parameter OUT_REG    = 0,
    parameter MAX_FANOUT = `MAX_FANOUT
) ( 
    input  wire             clk,
    input  wire             reset,

    // inputs    
    VX_dispatch_if.slave    dispatch_if [`ISSUE_WIDTH],

    // outputs
    VX_execute_if.master    execute_if [BLOCK_SIZE]

);
    `STATIC_ASSERT ((`NUM_THREADS == NUM_LANES  * (`NUM_THREADS / NUM_LANES)), ("invalid parameter"))
    localparam BLOCK_SIZE_W = `LOG2UP(BLOCK_SIZE);
    localparam NUM_PACKETS  = `NUM_THREADS / NUM_LANES;
    localparam PID_BITS     = `CLOG2(NUM_PACKETS);
    localparam PID_WIDTH    = `UP(PID_BITS);
    localparam BATCH_COUNT  = `ISSUE_WIDTH / BLOCK_SIZE;
    localparam BATCH_COUNT_W= `LOG2UP(BATCH_COUNT);
    localparam ISSUE_W      = `LOG2UP(`ISSUE_WIDTH);
    localparam IN_DATAW     = `UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + `INST_OP_BITS + `INST_MOD_BITS + 1 + 1 + 1 + `XLEN + `XLEN + `NR_BITS + `NT_WIDTH + (3 * `NUM_THREADS * `XLEN);
    localparam OUT_DATAW    = `UUID_WIDTH + `NW_WIDTH + NUM_LANES + `INST_OP_BITS + `INST_MOD_BITS + 1 + 1 + 1 + `XLEN + `XLEN + `NR_BITS + `NT_WIDTH + (3 * NUM_LANES * `XLEN) + PID_WIDTH + 1 + 1;
    localparam FANOUT_ENABLE= (`NUM_THREADS > (MAX_FANOUT + MAX_FANOUT/2));

    localparam DATA_TMASK_OFF = IN_DATAW - (`UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS);
    localparam DATA_REGS_OFF = 0;

    wire [`ISSUE_WIDTH-1:0] dispatch_valid;
    wire [`ISSUE_WIDTH-1:0][IN_DATAW-1:0] dispatch_data;
    wire [`ISSUE_WIDTH-1:0] dispatch_ready;

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        assign dispatch_valid[i] = dispatch_if[i].valid;
        assign dispatch_data[i] = dispatch_if[i].data;
        assign dispatch_if[i].ready = dispatch_ready[i];
    end
    
    wire [BLOCK_SIZE-1:0][ISSUE_W-1:0] issue_indices;
    wire [BLOCK_SIZE-1:0] block_ready;
    wire [BLOCK_SIZE-1:0][NUM_LANES-1:0] block_tmask;
    wire [BLOCK_SIZE-1:0][2:0][NUM_LANES-1:0][`XLEN-1:0] block_regs;
    wire [BLOCK_SIZE-1:0][PID_WIDTH-1:0] block_pid;
    wire [BLOCK_SIZE-1:0] block_sop;
    wire [BLOCK_SIZE-1:0] block_eop;
    wire [BLOCK_SIZE-1:0] block_done;

    wire batch_done = (& block_done);
    
    logic [BATCH_COUNT_W-1:0] batch_idx;
    // if (BATCH_COUNT != 1) begin
    //     always @(posedge clk) begin
    //         if (reset) begin
    //             batch_idx <= '0;
    //         end else begin
    //             batch_idx <= batch_idx + BATCH_COUNT_W'(batch_done);
    //         end
    //     end
    // end else begin
    //     assign batch_idx = 0;
    //     `UNUSED_VAR(batch_done)
    // end
    
    wire dispatch_any_valid;
    VX_lzc_rr #(
        .N       (`ISSUE_WIDTH)
    ) batch_select (
        .clk       (clk),
        .reset     (reset),
        .data_in   (dispatch_valid),
        .data_out  (batch_idx),
        .valid_out (dispatch_any_valid)
    );

    `STATIC_ASSERT ((BLOCK_SIZE == 1), ("dispatch_unit_sane only supports BLOCK_SIZE == 1 for now"))

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin

        wire [ISSUE_W-1:0] issue_idx = ISSUE_W'(batch_idx * BLOCK_SIZE) + ISSUE_W'(block_idx);
        assign issue_indices[block_idx] = issue_idx;

        wire valid_p, ready_p;

        assign valid_p = dispatch_valid[issue_idx];
        assign block_tmask[block_idx] = dispatch_data[issue_idx][DATA_TMASK_OFF +: `NUM_THREADS];
        assign block_regs[block_idx][0] = dispatch_data[issue_idx][DATA_REGS_OFF + 2 * `NUM_THREADS * `XLEN +: `NUM_THREADS * `XLEN];
        assign block_regs[block_idx][1] = dispatch_data[issue_idx][DATA_REGS_OFF + 1 * `NUM_THREADS * `XLEN +: `NUM_THREADS * `XLEN];
        assign block_regs[block_idx][2] = dispatch_data[issue_idx][DATA_REGS_OFF + 0 * `NUM_THREADS * `XLEN +: `NUM_THREADS * `XLEN];
        assign block_pid[block_idx]   = '0;
        assign block_sop[block_idx]   = 1'b1;
        assign block_eop[block_idx]   = 1'b1;
        assign block_ready[block_idx] = ready_p;
        assign block_done[block_idx]  = ~valid_p || ready_p;

        wire [ISSUE_ISW_W-1:0] isw;
        if (BATCH_COUNT != 1) begin
            if (BLOCK_SIZE != 1) begin
                assign isw = {batch_idx, BLOCK_SIZE_W'(block_idx)};
            end else begin
                assign isw = batch_idx;
            end
        end else begin
            assign isw = block_idx;
        end

        `RESET_RELAY(buf_out_reset, reset);

        wire [`NW_WIDTH-1:0] block_wid = wis_to_wid(dispatch_data[issue_idx][DATA_TMASK_OFF+`NUM_THREADS +: ISSUE_WIS_W], isw);

        VX_elastic_buffer #(
            .DATAW   (OUT_DATAW),
            .SIZE    (`OUT_REG_TO_EB_SIZE(OUT_REG)),
            .OUT_REG (`OUT_REG_TO_EB_REG(OUT_REG))
        ) buf_out (
            .clk       (clk),
            .reset     (buf_out_reset),
            .valid_in  (valid_p),
            .ready_in  (ready_p),
            .data_in   ({                
                dispatch_data[issue_idx][IN_DATAW-1 : DATA_TMASK_OFF+`NUM_THREADS+ISSUE_WIS_W],
                block_wid,
                block_tmask[block_idx],
                dispatch_data[issue_idx][DATA_TMASK_OFF-1 : DATA_REGS_OFF + 3 * `NUM_THREADS * `XLEN],
                block_regs[block_idx][0],
                block_regs[block_idx][1],
                block_regs[block_idx][2],     
                block_pid[block_idx],
                block_sop[block_idx],
                block_eop[block_idx]}),
            .data_out  (execute_if[block_idx].data),
            .valid_out (execute_if[block_idx].valid),
            .ready_out (execute_if[block_idx].ready)
        );
    end

    reg [`ISSUE_WIDTH-1:0] ready_in;
    always @(*) begin
        ready_in = 0;
        for (integer i = 0; i < BLOCK_SIZE; ++i) begin
            ready_in[issue_indices[i]] = block_ready[i] && block_eop[i];
        end
    end
    assign dispatch_ready = ready_in; 

endmodule
