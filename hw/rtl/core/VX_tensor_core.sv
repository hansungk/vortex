`ifdef EXT_T_ENABLE
`include "VX_fpu_define.vh"

module VX_tensor_core import VX_gpu_pkg::*; #(

) (
    input clk,
    input reset,

    VX_dispatch_if.slave dispatch_if [`ISSUE_WIDTH],
    VX_commit_if.master commit_if [`ISSUE_WIDTH]
);
    localparam BLOCK_SIZE = 1;
    localparam NUM_LANES  = `NUM_THREADS;
    // localparam PARTIAL_BW = (BLOCK_SIZE != `ISSUE_WIDTH) || (NUM_LANES != `NUM_THREADS);
    localparam PARTIAL_BW = 1;

    VX_execute_if #(
        .NUM_LANES (NUM_LANES)
    ) execute_if[BLOCK_SIZE]();

    `RESET_RELAY (dispatch_reset, reset);

    VX_dispatch_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_REG    (PARTIAL_BW ? 1 : 0)
    ) dispatch_unit (
        .clk        (clk),
        .reset      (dispatch_reset),
        .dispatch_if(dispatch_if),
        .execute_if (execute_if)
    );

    VX_commit_if #(
        .NUM_LANES (NUM_LANES)
    ) commit_block_if[BLOCK_SIZE]();

    `RESET_RELAY (commit_reset, reset);

    VX_gather_unit #(
        .BLOCK_SIZE (BLOCK_SIZE),
        .NUM_LANES  (NUM_LANES),
        .OUT_REG    (PARTIAL_BW ? 3 : 0) // FIXME: why 3?
    ) gather_unit (
        .clk           (clk),
        .reset         (commit_reset),
        .commit_in_if  (commit_block_if),
        .commit_out_if (commit_if)
    );

    for (genvar block_idx = 0; block_idx < BLOCK_SIZE; ++block_idx) begin
        VX_tensor_core_warp #(
            .ISW(1) // FIXME: not block_idx
        ) tensor_core (
            .clk(clk),
            .reset(reset),

            .execute_if(execute_if[block_idx]),
            .commit_if(commit_block_if[block_idx])
        );
    end
    
endmodule

module VX_tensor_core_warp import VX_gpu_pkg::*; #(
    parameter ISW
) (
    input clk,
    input reset,

    VX_execute_if.slave execute_if,
    VX_commit_if.master commit_if
);
    localparam NUM_OCTETS = (`NUM_THREADS / 8);
    // offet in the lane numbers that get mapped to the two threadgroups in an
    // octet. E.g. two tgs map lane 0-3 and lane 16-19 -> 16
    // FIXME: not sure this is the right logic.  just filling in what works
    localparam LANE_OFFSET_THREADGROUP = (4 * NUM_OCTETS);

    wire [1:0] step = 2'(execute_if.data.op_type);
    logic [NUM_OCTETS-1:0] octet_results_valid;
    logic [NUM_OCTETS-1:0] octet_results_ready;
    logic [NUM_OCTETS-1:0] octet_operands_ready;
    // FIXME: should be NUM_LANES?
    logic [`NUM_THREADS-1:0][`XLEN-1:0] wb_data_0;
    logic [`NUM_THREADS-1:0][`XLEN-1:0] wb_data_1;
    
    assign execute_if.ready = &octet_operands_ready;

`ifdef EXT_T_ENABLE
    for (genvar i = 0; i < NUM_OCTETS; ++i) begin
`else
    for (genvar i = 0; i < 0; ++i) begin
`endif
        // lane-to-octet mapping; see figure 13 of the paper
        wire [7:0][31:0] octet_A = {
            execute_if.data.rs1_data[LANE_OFFSET_THREADGROUP + 4*i +: 4], execute_if.data.rs1_data[4*i +: 4]
        };
        wire [7:0][31:0] octet_B = {
            execute_if.data.rs2_data[LANE_OFFSET_THREADGROUP + 4*i +: 4], execute_if.data.rs2_data[4*i +: 4]
        };
        wire [7:0][31:0] octet_C = {
            execute_if.data.rs3_data[LANE_OFFSET_THREADGROUP + 4*i +: 4], execute_if.data.rs3_data[4*i +: 4]
        };

        logic [3:0][3:0][31:0] octet_D;
        logic result_valid;
        logic result_ready;
        VX_tensor_octet #(
            .ISW(ISW),
            .OCTET(i)
        ) octet (
            .clk(clk),
            .reset(reset),

            .A_in(octet_A),
            .B_in(octet_B),
            .C_in(octet_C),
            .operands_valid(execute_if.valid),
            .operands_ready(octet_operands_ready[i]),

            .step(step),

            .D_out(octet_D),
            .result_valid(result_valid),
            .result_ready(result_ready)
        );

        // these should always be in lockstep
        assign octet_results_valid[i] = result_valid;
        assign result_ready = octet_results_ready[i];

        // each octet produces 4x4 output partial sum, but the 8 lanes mapped
        // to the octet can only do 8 fp32 writeback at a time; so we need to
        // split writeback over two cycles
        assign wb_data_0[4*i+0] = octet_D[0][0];
        assign wb_data_0[4*i+1] = octet_D[1][0];
        assign wb_data_0[4*i+2] = octet_D[0][2];
        assign wb_data_0[4*i+3] = octet_D[1][2];

        assign wb_data_1[4*i+0] = octet_D[0][1];
        assign wb_data_1[4*i+1] = octet_D[1][1];
        assign wb_data_1[4*i+2] = octet_D[0][3];
        assign wb_data_1[4*i+3] = octet_D[1][3];

        assign wb_data_0[4*i+LANE_OFFSET_THREADGROUP+0] = octet_D[2][0];
        assign wb_data_0[4*i+LANE_OFFSET_THREADGROUP+1] = octet_D[3][0];
        assign wb_data_0[4*i+LANE_OFFSET_THREADGROUP+2] = octet_D[2][2];
        assign wb_data_0[4*i+LANE_OFFSET_THREADGROUP+3] = octet_D[3][2];

        assign wb_data_1[4*i+LANE_OFFSET_THREADGROUP+0] = octet_D[2][1];
        assign wb_data_1[4*i+LANE_OFFSET_THREADGROUP+1] = octet_D[3][1];
        assign wb_data_1[4*i+LANE_OFFSET_THREADGROUP+2] = octet_D[2][3];
        assign wb_data_1[4*i+LANE_OFFSET_THREADGROUP+3] = octet_D[3][3];
    end
    
    /* commit_if.data_t parts that we need to keep around:
        - uuid
        - wid
        - tmask
        - PC
        - wb
        - rd
    */

    localparam DATAW = `UUID_WIDTH + `NW_WIDTH + `NUM_THREADS + `XLEN + 1 + `NR_BITS;
    
    wire execute_if_fire = execute_if.valid && execute_if.ready;
    wire commit_if_fire = commit_if.valid && commit_if.ready;
    wire [DATAW-1:0] execute_if_data_enq = {
        execute_if.data.uuid, 
        execute_if.data.wid,
        execute_if.data.tmask, 
        execute_if.data.PC, 
        execute_if.data.wb, 
        execute_if.data.rd
    };

    wire [DATAW-1:0] execute_if_data_deq;

    // this is probably a little oversized
    VX_fifo_queue #(
        .DATAW(DATAW),
        .DEPTH(16)
    ) pending_uops (
        .clk(clk),
        .reset(reset),    
        .push(execute_if_fire),
        .pop(commit_if_fire),
        .data_in(execute_if_data_enq),
        .data_out(execute_if_data_deq),
        `UNUSED_PIN(empty),      
        `UNUSED_PIN(alm_empty),
        `UNUSED_PIN(full), // should be impossible to overflow            
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    logic subcommit, subcommit_n;
    wire all_valid = (& octet_results_valid);
    assign commit_if.valid = all_valid;

    localparam COMMIT_DATAW = `UUID_WIDTH + `NW_WIDTH + `NUM_THREADS + `XLEN + 1 + `NR_BITS + (`NUM_THREADS * `XLEN) + 1 + 1 + 1;
    wire [COMMIT_DATAW-1:0] commit_if_data = {
        execute_if_data_deq, /* uuid ~ rd */
        subcommit == 1'b0 ? wb_data_0 : wb_data_1, /* data */
        1'b0, /* pid */
        1'b1, /* sop */
        1'b1  /* eop */
    };

    assign commit_if.data = commit_if_data;

    always @(*) begin
        subcommit_n = commit_if_fire ? ~subcommit : subcommit;
        if (commit_if_fire && subcommit == 1'b1) begin
            octet_results_ready = '1;
        end
        else begin
            octet_results_ready = '0;
        end  
    end

    always @(posedge clk) begin
        if (reset) begin
            subcommit <= '0;
        end
        else begin
            subcommit <= subcommit_n;
        end
    end
    
endmodule

module VX_tensor_octet #(
    parameter ISW,
    parameter OCTET
) (
    input clk,
    input reset,

    input [7:0][31:0] A_in,
    input [7:0][31:0] B_in,
    input [7:0][31:0] C_in,
    input operands_valid, // we have to backpressure due to there potentially being contention over commit
    output operands_ready,

    input [1:0] step,

    output [3:0][3:0][31:0] D_out,
    output result_valid,
    input result_ready
);
    // 512 bits/octet * 4 octets per warp
    logic [3:0][31:0] A_buffer, A_buffer_n;
    logic [3:0][31:0] B_buffer, B_buffer_n;
    logic [7:0][31:0] C_buffer, C_buffer_n;

    // half the inputs are buffered, half are not (instead coming straight
    // from operand bus) unlike the real tensor core.
    // the banks are only 32 bit rather than 64 bit (a pair of fp32 regs).
    logic [3:0][31:0] A_half;
    logic [3:0][31:0] B_half;
    logic [7:0][31:0] C_half;
    always @(*) begin
        // note that not all lanes participate at every step
        case (step)
            2'b00: begin
                // Two A_in segments correspond to two 2x2 subtiles of A read
                // by two threadgroups: [0:2,0:2] and [4:6,0:2] in Step 0 of
                // Figure 10(b).  B_in OTOH is shared by two threadgroups.
                // Note k-dimension is shrunk from 4 to 2.
                A_half = { A_in[5:4], A_in[1:0] };
                B_half = B_in[3:0];
            end
            2'b01: begin
                A_half = { A_in[7:6], A_in[3:2] };
                B_half = B_in[3:0];
            end
            2'b10: begin
                A_half = { A_in[5:4], A_in[1:0] };
                B_half = B_in[7:4];
            end
            2'b11: begin
                A_half = { A_in[7:6], A_in[3:2] };
                B_half = B_in[7:4];
            end
        endcase
        C_half = C_in;
    end

    logic substep;
    wire substep_n = (operands_ready && operands_valid) ? ~substep : substep;

    always @(*) begin
        A_buffer_n = A_buffer;
        B_buffer_n = B_buffer;
        C_buffer_n = C_buffer;
        
        if (substep == 1'b0) begin
            A_buffer_n = A_half;
            B_buffer_n = B_half;
            C_buffer_n = C_half;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            A_buffer <= '0;
            B_buffer <= '0;
            C_buffer <= '0;
            substep <= '0;
        end
        else begin
            A_buffer <= A_buffer_n;
            B_buffer <= B_buffer_n;
            C_buffer <= C_buffer_n;
            substep <= substep_n;
        end
    end

    wire stall = result_valid && ~result_ready;
    assign operands_ready = ~stall;

    // A is 4x2 fp32 matrix
    wire [3:0][1:0][31:0] A_tile = {
        { A_half[3], A_buffer[3] },
        { A_half[2], A_buffer[2] },
        { A_half[1], A_buffer[1] },
        { A_half[0], A_buffer[0] }
    };
    // B is 2x4 fp32 matrix
    wire [1:0][3:0][31:0] B_tile = {
        B_half, B_buffer
    };
    // C is 4x4 fp32 matrix
    logic [3:0][3:0][31:0] C_tile;
    
    always @(*) begin
        C_tile = {
            C_half[7], C_buffer[7], C_half[5], C_buffer[5],
            C_half[6], C_buffer[6], C_half[4], C_buffer[4],
            C_half[3], C_buffer[3], C_half[1], C_buffer[1],
            C_half[2], C_buffer[2], C_half[0], C_buffer[0]
        };
    end 

    wire do_hmma = (substep == 1'b1 && operands_valid && operands_ready);

    // this does (m,n,k)=(4,4,2) matmul, modeling compute of a single octet
    VX_tensor_dpu #(
        .ISW(ISW),
        .OCTET(OCTET)
    ) dpu (
        .clk(clk),
        .reset(reset),

        .stall(stall),
        
        .valid_in(do_hmma),
        .A_tile(A_tile),
        .B_tile(B_tile),
        .C_tile(C_tile),

        .valid_out(result_valid),
        .D_tile(D_out)
    );
endmodule
`endif
