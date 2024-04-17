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

`ifdef GPR_DUPLICATED

module VX_operands_dup import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0,
    parameter CACHE_ENABLE = 0
) (
    input wire              clk,
    input wire              reset,

    VX_writeback_if.slave   writeback_if [`ISSUE_WIDTH],
    VX_ibuffer_if.slave     scoreboard_if [`ISSUE_WIDTH],
    VX_operands_if.master   operands_if [`ISSUE_WIDTH]
);
    `UNUSED_PARAM (CORE_ID)
    localparam DATAW = `UUID_WIDTH + ISSUE_WIS_W + `NUM_THREADS + `XLEN + 1 + `EX_BITS + `INST_OP_BITS + `INST_MOD_BITS + 1 + 1 + `XLEN + `NR_BITS;
    localparam RAM_ADDRW = `LOG2UP(`NUM_REGS * ISSUE_RATIO);

    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        // NOTE(hansung): toggle_buffer is 1-reg pipe without flow, halving
        // throughput.  Wouldn't this cap overall IPC?  Or OK as long as
        // ISSUE_WIDTH > 1?
        VX_stream_buffer #(
            .DATAW (DATAW)
        ) staging_buffer (
            .clk       (clk),
            .reset     (reset),
            .valid_in  (scoreboard_if[i].valid),
            .data_in   ({
                scoreboard_if[i].data.uuid,
                scoreboard_if[i].data.wis,
                scoreboard_if[i].data.tmask,
                scoreboard_if[i].data.PC, 
                scoreboard_if[i].data.wb,
                scoreboard_if[i].data.ex_type,
                scoreboard_if[i].data.op_type,
                scoreboard_if[i].data.op_mod,
                scoreboard_if[i].data.use_PC,
                scoreboard_if[i].data.use_imm,
                scoreboard_if[i].data.imm,
                scoreboard_if[i].data.rd
            }),
            .ready_in  (scoreboard_if[i].ready),
            .valid_out (operands_if[i].valid),
            .data_out  ({
                operands_if[i].data.uuid,
                operands_if[i].data.wis,
                operands_if[i].data.tmask,
                operands_if[i].data.PC, 
                operands_if[i].data.wb,
                operands_if[i].data.ex_type,
                operands_if[i].data.op_type,
                operands_if[i].data.op_mod,
                operands_if[i].data.use_PC,
                operands_if[i].data.use_imm,
                operands_if[i].data.imm,
                operands_if[i].data.rd
            }),
            .ready_out (operands_if[i].ready)
        );

        wire [`NUM_THREADS-1:0][`XLEN-1:0] rs1_data;
        wire [`NUM_THREADS-1:0][`XLEN-1:0] rs2_data;
        wire [`NUM_THREADS-1:0][`XLEN-1:0] rs3_data;

        for (genvar j = 0; j < `NUM_THREADS; ++j) begin
            VX_stream_buffer #(
                .DATAW (`XLEN + `XLEN + `XLEN)
            ) staging_data_buffer (
                .clk       (clk),
                .reset     (reset),
                .valid_in  (scoreboard_if[i].valid),
                .data_in   ({
                   rs1_data[j], rs2_data[j], rs3_data[j]
                }),
                `UNUSED_PIN (ready_in),
                `UNUSED_PIN (valid_out),
                .data_out  ({
                   operands_if[i].data.rs1_data[j],
                   operands_if[i].data.rs2_data[j],
                   operands_if[i].data.rs3_data[j]
                }),
                .ready_out (operands_if[i].ready)
            );
        end

        // GPR banks

        wire [RAM_ADDRW-1:0] gpr_rd_addr_rs1;
        wire [RAM_ADDRW-1:0] gpr_rd_addr_rs2;
        wire [RAM_ADDRW-1:0] gpr_rd_addr_rs3;
        wire [RAM_ADDRW-1:0] gpr_wr_addr;
        if (ISSUE_WIS != 0) begin
            assign gpr_wr_addr = {writeback_if[i].data.wis, writeback_if[i].data.rd};
            assign gpr_rd_addr_rs1 = {scoreboard_if[i].data.wis, scoreboard_if[i].data.rs1};
            assign gpr_rd_addr_rs2 = {scoreboard_if[i].data.wis, scoreboard_if[i].data.rs2};
            assign gpr_rd_addr_rs3 = {scoreboard_if[i].data.wis, scoreboard_if[i].data.rs3};
            // always @(posedge clk) begin
            //     if (reset) begin
            //         gpr_rd_addr_rs1 <= '0;
            //         gpr_rd_addr_rs2 <= '0;
            //         gpr_rd_addr_rs3 <= '0;
            //     end else begin
            //         // if (!(operands_if[i].valid && !operands_if[i].ready)) begin
            //         if (scoreboard_if[i].valid && scoreboard_if[i].ready) begin
            //             gpr_rd_addr_rs1 <= {scoreboard_if[i].data.wis, scoreboard_if[i].data.rs1};
            //             gpr_rd_addr_rs2 <= {scoreboard_if[i].data.wis, scoreboard_if[i].data.rs2};
            //             gpr_rd_addr_rs3 <= {scoreboard_if[i].data.wis, scoreboard_if[i].data.rs3};
            //         end
            //     end
            // end
        end else begin
            assign gpr_wr_addr = writeback_if[i].data.rd;
            assign gpr_rd_addr_rs1 = scoreboard_if[i].data.rs1;
            assign gpr_rd_addr_rs2 = scoreboard_if[i].data.rs2;
            assign gpr_rd_addr_rs3 = scoreboard_if[i].data.rs3;
            // always @(posedge clk) begin
            //     if (reset) begin
            //         gpr_rd_addr_rs1 <= '0;
            //         gpr_rd_addr_rs2 <= '0;
            //         gpr_rd_addr_rs3 <= '0;
            //     end else begin
            //         // if (!(operands_if[i].valid && !operands_if[i].ready)) begin
            //         if (scoreboard_if[i].valid && scoreboard_if[i].ready) begin
            //             gpr_rd_addr_rs1 <= scoreboard_if[i].data.rs1;
            //             gpr_rd_addr_rs2 <= scoreboard_if[i].data.rs2;
            //             gpr_rd_addr_rs3 <= scoreboard_if[i].data.rs3;
            //         end
            //     end
            // end
        end
        
    `ifdef GPR_RESET
        reg wr_enabled = 0;
        always @(posedge clk) begin
            if (reset) begin
                wr_enabled <= 1;
            end
        end
    `endif

        for (genvar j = 0; j < `NUM_THREADS; ++j) begin
            VX_dp_ram #(
                .DATAW (`XLEN),
                .SIZE (`NUM_REGS * ISSUE_RATIO),
            `ifdef GPR_RESET
                .INIT_ENABLE (1),
                .INIT_VALUE (0),
            `endif
                .NO_RWCHECK (1)
            ) gpr_ram_rs1 (
                .clk   (clk),
                .read  (1'b1),
                `UNUSED_PIN (wren),
            `ifdef GPR_RESET
                .write (wr_enabled && writeback_if[i].valid && writeback_if[i].data.tmask[j]),
            `else
                .write (writeback_if[i].valid && writeback_if[i].data.tmask[j]),
            `endif              
                .waddr (gpr_wr_addr),
                .wdata (writeback_if[i].data.data[j]),
                .raddr (gpr_rd_addr_rs1),
                .rdata (rs1_data[j])
            );

            VX_dp_ram #(
                .DATAW (`XLEN),
                .SIZE (`NUM_REGS * ISSUE_RATIO),
            `ifdef GPR_RESET
                .INIT_ENABLE (1),
                .INIT_VALUE (0),
            `endif
                .NO_RWCHECK (1)
            ) gpr_ram_rs2(
                .clk   (clk),
                .read  (1'b1),
                `UNUSED_PIN (wren),
            `ifdef GPR_RESET
                .write (wr_enabled && writeback_if[i].valid && writeback_if[i].data.tmask[j]),
            `else
                .write (writeback_if[i].valid && writeback_if[i].data.tmask[j]),
            `endif              
                .waddr (gpr_wr_addr),
                .wdata (writeback_if[i].data.data[j]),
                .raddr (gpr_rd_addr_rs2),
                .rdata (rs2_data[j])
            );

            VX_dp_ram #(
                .DATAW (`XLEN),
                .SIZE (`NUM_REGS * ISSUE_RATIO),
            `ifdef GPR_RESET
                .INIT_ENABLE (1),
                .INIT_VALUE (0),
            `endif
                .NO_RWCHECK (1)
            ) gpr_ram_rs3 (
                .clk   (clk),
                .read  (1'b1),
                `UNUSED_PIN (wren),
            `ifdef GPR_RESET
                .write (wr_enabled && writeback_if[i].valid && writeback_if[i].data.tmask[j]),
            `else
                .write (writeback_if[i].valid && writeback_if[i].data.tmask[j]),
            `endif              
                .waddr (gpr_wr_addr),
                .wdata (writeback_if[i].data.data[j]),
                .raddr (gpr_rd_addr_rs3),
                .rdata (rs3_data[j])
            );
        end
    end

endmodule

`endif
