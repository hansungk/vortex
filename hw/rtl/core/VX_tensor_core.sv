`include "VX_fpu_define.vh"

module VX_tensor_core #(

) (
    input clk,
    input reset,

    VX_dispatch_if.slave dispatch_if [`ISSUE_WIDTH],
    VX_commit_if.master commit_if [`ISSUE_WIDTH]
);
    `STATIC_ASSERT(`NUM_THREADS == 32, ("tensor core requires # of threads in a warp to be 32 (try running w/ CONFIGS=\"-DNUM_THREADS=32\")"));
    
    for (genvar i = 0; i < `ISSUE_WIDTH; ++i) begin
        VX_tensor_core_warp #(
            .ISW(i)
        ) tensor_core (
            .clk(clk),
            .reset(reset),

            .dispatch_if(dispatch_if[i]),
            .commit_if(commit_if[i])
        );
    end
    
endmodule

module VX_tensor_core_warp import VX_gpu_pkg::*; #(
    parameter ISW
) (
    input clk,
    input reset,

    VX_dispatch_if.slave dispatch_if,
    VX_commit_if.master commit_if
);
    wire [1:0] step = 2'(dispatch_if.data.op_type);
    logic [3:0] octet_results_valid;
    logic [3:0] octet_results_ready;
    logic [3:0] octet_operands_ready;
    logic [`NUM_THREADS-1:0][`XLEN-1:0] wb_data_0;
    logic [`NUM_THREADS-1:0][`XLEN-1:0] wb_data_1;
    
    assign dispatch_if.ready = &octet_operands_ready;

    for (genvar i = 0; i < 4/*octets*/; ++i) begin
        // lane-to-octet mapping; see figure 13 of the paper
        wire [7:0][31:0] octet_A = {
            dispatch_if.data.rs1_data[16+4*i +: 4], dispatch_if.data.rs1_data[4*i +: 4]
        };
        wire [7:0][31:0] octet_B = {
            dispatch_if.data.rs2_data[16+4*i +: 4], dispatch_if.data.rs2_data[4*i +: 4]
        };
        wire [7:0][31:0] octet_C = {
            dispatch_if.data.rs3_data[16+4*i +: 4], dispatch_if.data.rs3_data[4*i +: 4]
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
            .operands_valid(dispatch_if.valid),
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

        assign wb_data_0[4*i+16+0] = octet_D[2][0];
        assign wb_data_0[4*i+16+1] = octet_D[3][0];
        assign wb_data_0[4*i+16+2] = octet_D[2][2];
        assign wb_data_0[4*i+16+3] = octet_D[3][2];

        assign wb_data_1[4*i+16+0] = octet_D[2][1];
        assign wb_data_1[4*i+16+1] = octet_D[3][1];
        assign wb_data_1[4*i+16+2] = octet_D[2][3];
        assign wb_data_1[4*i+16+3] = octet_D[3][3];
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
    
    wire dispatch_if_fire = dispatch_if.valid && dispatch_if.ready;
    wire commit_if_fire = commit_if.valid && commit_if.ready;
    wire [DATAW-1:0] dispatch_if_data_enq = {
        dispatch_if.data.uuid, 
        wis_to_wid(dispatch_if.data.wis, ISW), 
        dispatch_if.data.tmask, 
        dispatch_if.data.PC, 
        dispatch_if.data.wb, 
        dispatch_if.data.rd
    };

    wire [DATAW-1:0] dispatch_if_data_deq;

    // this is probably a little oversized
    VX_fifo_queue #(
        .DATAW(DATAW),
        .DEPTH(16)
    ) pending_uops (
        .clk(clk),
        .reset(reset),    
        .push(dispatch_if_fire),
        .pop(commit_if_fire),
        .data_in(dispatch_if_data_enq),
        .data_out(dispatch_if_data_deq),
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
        dispatch_if_data_deq, /* uuid ~ rd */
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
