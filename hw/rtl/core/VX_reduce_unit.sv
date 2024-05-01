`include "VX_define.vh"
`include "VX_platform.vh"


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

`include "VX_platform.vh"

module VX_reduce_ext #(    
    parameter DATAW_IN   = 1,
    parameter DATAW_OUT  = DATAW_IN,
    parameter N          = 1
) (
    input wire [N-1:0][DATAW_IN-1:0] data_in,
    input wire [N-1:0]               mask,
    input wire [`INST_RED_BITS-1:0]  op_type,
    output wire [DATAW_OUT-1:0]      data_out
);
    if (N == 1) begin
        `UNUSED_VAR(op_type)
        `UNUSED_VAR(mask)
        assign data_out = DATAW_OUT'(data_in[0]);
    end else begin
        localparam int N_A = N / 2;
        localparam int N_B = N - N_A;

        wire [N_A-1:0][DATAW_IN-1:0] in_A;
        wire [N_B-1:0][DATAW_IN-1:0] in_B;
        wire [DATAW_OUT-1:0] out_A, out_B;

        wire [N_A-1:0] mask_A;
        wire [N_B-1:0] mask_B;
        wire any_A, any_B;

        for (genvar i = 0; i < N_A; i++) begin
            assign in_A[i] = data_in[i];
        end

        for (genvar i = 0; i < N_B; i++) begin
            assign in_B[i] = data_in[N_A + i];
        end

        assign mask_A = mask[N_A-1:0];
        assign mask_B = mask[N-1:N_A];
        assign any_A = |mask_A;
        assign any_B = |mask_B;

        VX_reduce_ext #(
            .DATAW_IN  (DATAW_IN), 
            .DATAW_OUT (DATAW_OUT),
            .N  (N_A)
        ) reduce_A (
            .data_in  (in_A),
            .mask(mask_A),
            .op_type(op_type),
            .data_out (out_A)
        );

        VX_reduce_ext #(
            .DATAW_IN  (DATAW_IN), 
            .DATAW_OUT (DATAW_OUT),
            .N  (N_B)
        ) reduce_B (
            .data_in  (in_B),
            .mask(mask_B),
            .op_type(op_type),
            .data_out (out_B)
        );

        logic [DATAW_OUT-1:0] _data_out;
        
        always @(*) begin
            case (op_type)
                `INST_RED_ADD: _data_out = out_A + out_B;
                `INST_RED_ADDU: _data_out = out_A + out_B;
                `INST_RED_MIN: _data_out = ($signed(out_A) < $signed(out_B)) ? out_A : out_B;
                `INST_RED_MINU: _data_out = (out_A < out_B) ? out_A : out_B;
                `INST_RED_MAX: _data_out = ($signed(out_A) < $signed(out_B)) ? out_B : out_A;
                `INST_RED_MAXU: _data_out = (out_A < out_B) ? out_B : out_A;
                `INST_RED_AND: _data_out = out_A & out_B;
                `INST_RED_OR: _data_out = out_A | out_B;
                `INST_RED_XOR: _data_out = out_A ^ out_B;
                default: _data_out = out_A;
            endcase
        end
        
        // if both sides are masked out, then it doesn't matter what we output
        assign data_out = (any_A && any_B) ? _data_out : (any_A ? out_A : out_B);

    end
  
endmodule

module VX_reduce_unit #(
    parameter CORE_ID = 0,
    parameter NUM_LANES = 1
) (
    input wire clk,
    input wire reset,

    VX_execute_if.slave execute_if,
    VX_commit_if.master commit_if
);
    `UNUSED_PARAM(CORE_ID)

    localparam NUM_PACKETS = `NUM_THREADS / NUM_LANES;
    localparam PID_BITS    = `CLOG2(`NUM_THREADS / NUM_LANES);
    localparam PID_WIDTH   = `UP(PID_BITS);

    logic [`XLEN-1:0] accumulator, accumulator_n, reduced_accumulator;
    wire [(NUM_LANES * `XLEN)-1:0] broadcasted_accumulator;

    assign broadcasted_accumulator = {NUM_LANES{accumulator}};

    wire eop;
    wire [NUM_LANES-1:0][`XLEN-1:0] data_in;
    wire [`XLEN-1:0] data_out;
    
    assign eop = execute_if.data.eop;
    assign data_in = execute_if.data.rs1_data;

    logic execute_if_valid;
    logic execute_if_ready;
    logic commit_if_valid;
    logic commit_if_ready;

    wire execute_if_fire;
    wire commit_if_fire;

    assign execute_if_valid = execute_if.valid;
    assign execute_if.ready = execute_if_ready;

    assign execute_if_fire = execute_if.ready && execute_if.valid;
    assign commit_if_fire = commit_if_ready && commit_if_valid;

    logic store_tmask_pid;
    logic read_tmask_pid;
    wire [PID_WIDTH-1:0] stored_pid;
    wire [NUM_LANES-1:0] stored_tmask;
    wire stored_sop;
    wire stored_eop;

    logic [PID_BITS:0] size, size_n;
    
    // 1. idle state - wait for execute_if to be valid
    // 2. accumulate - continue accumulating until eop, store packet id + thread mask for broadcast phase
    // 3. broadcast - broadcast to rds
    localparam IDLE = 2'b00;
    localparam ACCUMULATE = 2'b01;
    localparam BROADCAST = 2'b10;
    localparam FINISH = 2'b11;

    logic [1:0] state, state_n;

    always @(*) begin
        state_n = state;
        accumulator_n = accumulator;
        execute_if_ready = '0;
        commit_if_valid = '0;
        store_tmask_pid = '0;
        read_tmask_pid = '0;
        size_n = store_tmask_pid ? size + 1 : (read_tmask_pid ? size - 1 : size);

        case (state)
            IDLE: begin
                if (execute_if_valid) begin
                    accumulator_n = data_out;
                    store_tmask_pid = '1;
                    if (eop) begin
                        state_n = BROADCAST;
                    end
                    else begin
                        execute_if_ready = '1;
                        state_n = ACCUMULATE;
                    end
                end
            end
            ACCUMULATE: begin
                execute_if_ready = '1;
                if (eop) begin
                    execute_if_ready = '0;
                    state_n = BROADCAST;
                end
                if (eop || execute_if_fire) begin
                    accumulator_n = reduced_accumulator;
                    store_tmask_pid = '1;
                end
            end
            BROADCAST: begin
                execute_if_ready = '0;
                commit_if_valid = '1;
                
                if (commit_if_fire) begin
                    read_tmask_pid = '1;
                end
                if (size_n == '0) begin
                    state_n = FINISH;
                end
            end
            FINISH: begin
                execute_if_ready = '1;
                if (execute_if_fire) begin
                    state_n = IDLE;
                end
            end
        endcase
    end

    always @(posedge clk) begin
        if (reset) begin
            accumulator <= '0;
            state <= IDLE;
            size <= '0;
        end
        else begin
            accumulator <= accumulator_n;
            state <= state_n;
            size <= size_n;
        end
    end

    VX_reduce_ext #(    
        .DATAW_IN(`XLEN),
        .N(NUM_LANES)
    ) reducer (
        .data_in(data_in),
        .mask(execute_if.data.tmask),
        .op_type(execute_if.data.op_type),
        .data_out(data_out)
    );

    VX_reduce_ext #(
        .DATAW_IN(`XLEN),
        .N(2)
    ) accumulator_reducer (
        .data_in({accumulator, data_out}),
        .mask(2'b11),
        .op_type(execute_if.data.op_type),
        .data_out(reduced_accumulator)
    );

    VX_elastic_buffer #(
        .DATAW(NUM_LANES + PID_WIDTH + 1 + 1),
        .SIZE(NUM_PACKETS),
    ) tmask_pid_store (
        .clk(clk),
        .reset(reset),

        .valid_in(store_tmask_pid),
        `UNUSED_PIN(ready_in),
        .data_in({execute_if.data.tmask, execute_if.data.pid, execute_if.data.sop, execute_if.data.eop}),
    
        .data_out({stored_tmask, stored_pid, stored_sop, stored_eop}),
        .ready_out(read_tmask_pid),
        `UNUSED_PIN(valid_out)
    );

    VX_elastic_buffer #(
        .DATAW(`UUID_WIDTH + `NW_WIDTH + NUM_LANES + `XLEN + 1 + `NR_BITS + (`XLEN * NUM_LANES) + PID_WIDTH + 1 + 1)
    ) output_buffer (
        .clk(clk),
        .reset(reset),
        .valid_in(commit_if_valid),
        .ready_in(commit_if_ready),
        .data_in({execute_if.data.uuid, execute_if.data.wid, stored_tmask, execute_if.data.PC, execute_if.data.wb, execute_if.data.rd, broadcasted_accumulator, stored_pid, stored_sop, stored_eop}),

        .data_out({commit_if.data.uuid, commit_if.data.wid, commit_if.data.tmask, commit_if.data.PC, commit_if.data.wb, commit_if.data.rd, commit_if.data.data, commit_if.data.pid, commit_if.data.sop, commit_if.data.eop}),
        .ready_out(commit_if.ready),
        .valid_out(commit_if.valid)
    );

endmodule
