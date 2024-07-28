`ifdef EXT_T_ENABLE
`include "VX_fpu_define.vh"

module VX_tensor_dpu #(
    parameter ISW,
    parameter OCTET,
    // @perf: has big impact on throughput.  A rule of thumb is to set it to
    // the pipeline length of FEDPs in order to make sure there are enough
    // entries to fully saturate the pipeline, but this is still rough
    parameter ISSUE_QUEUE_DEPTH = `LATENCY_HMMA
) (
    input clk,
    input reset,

    input valid_in,
    output ready_in,
    // [rows][cols][dtype]
    input [3:0][1:0][31:0] A_tile,
    input [1:0][3:0][31:0] B_tile,
    input [3:0][3:0][31:0] C_tile,
    input [`NW_WIDTH-1:0]  wid,

    output valid_out,
    input  ready_out,
    output [3:0][3:0][31:0] D_tile,
    output [`NW_WIDTH-1:0]  D_wid
);
    // logic [3:0][3:0][31:0] result_hmma;

    // always @(*) begin
    //     dpi_hmma(valid_in, A_tile, B_tile, C_tile, result_hmma);
    // end

    // // fixed-latency queue
    // VX_shift_register #(
    //     .DATAW  (1 + $bits(wid)/* + $bits(D_tile)*/),
    //     .DEPTH  (`LATENCY_HMMA + 1),
    //     .RESETW (1)
    // ) shift_reg (
    //     .clk      (clk),
    //     .reset    (reset),
    //     .enable   (ready_out),
    //     .data_in  ({valid_in && ready_in, wid  /*, result_hmma*/}),
    //     .data_out ({valid_out,            D_wid/*, D_tile     */})
    // );

    // ready as soon as valid_out
    // assign ready_in = valid_out;

    // fully pipelined; ready_in is coupled to ready_out by immediately
    // stalling
    // assign ready_in = ready_out;

    wire synced_fire;
    assign synced_fire = valid_in && ready_in;

    wire [1:0] threadgroup_valids;
    wire [1:0] threadgroup_readys;
    // B_tile is shared across the two threadgroups; see Figure 13
    VX_tensor_threadgroup #(
        .ISSUE_QUEUE_DEPTH(ISSUE_QUEUE_DEPTH)
    ) threadgroup_0 (
        .clk   (clk),
        .reset (reset),
        .valid_in  (synced_fire),
        .ready_in  (threadgroup_readys[0]),
        .stall     (!ready_out),
        .A_frag    (A_tile[1:0]),
        .B_frag    (B_tile),
        .C_frag    (C_tile[1:0]),
        .valid_out (threadgroup_valids[0]),
        .D_frag    (D_tile[1:0])
    );
    VX_tensor_threadgroup #(
        .ISSUE_QUEUE_DEPTH(ISSUE_QUEUE_DEPTH)
    ) threadgroup_1 (
        .clk   (clk),
        .reset (reset),
        .valid_in  (synced_fire),
        .ready_in  (threadgroup_readys[1]),
        .stall     (!ready_out),
        .A_frag    (A_tile[3:2]),
        .B_frag    (B_tile),
        .C_frag    (C_tile[3:2]),
        .valid_out (threadgroup_valids[1]),
        .D_frag    (D_tile[3:2])
    );

    wire empty;
    wire full;
    wire enq = valid_in && ready_in;
    wire deq = valid_out && ready_out;

    assign ready_in  = &(threadgroup_readys) && !full;
    assign valid_out = &(threadgroup_valids);

    // need to pass along warp id's to do multithreading
    VX_fifo_queue #(
        .DATAW   ($bits(wid)),
        // @perf: seems to require deeper depth than the FEDP issue queues to
        // not cause stalls.
        .DEPTH   (2 * ISSUE_QUEUE_DEPTH)
    ) wid_queue (
        .clk   (clk),
        .reset (reset),
        .push      (enq),
        .pop       (deq),
        .data_in   (wid),
        .data_out  (D_wid),
        .empty     (empty),
        `UNUSED_PIN(alm_empty),
        .full      (full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    `RUNTIME_ASSERT(reset || !(deq && empty),
                    ("dequeueing from empty warp id queue!"))
endmodule

// does (m,n,k) = (2,4,2) matmul compute over 2 cycles.
// matches Figure 10(b) of the paper.
module VX_tensor_threadgroup #(
    parameter ISSUE_QUEUE_DEPTH
) (
    input clk,
    input reset,

    input valid_in,
    output ready_in,
    input stall,
    input [1:0][1:0][31:0] A_frag,
    input [1:0][3:0][31:0] B_frag,
    input [1:0][3:0][31:0] C_frag,

    output valid_out,
    output [1:0][3:0][31:0] D_frag
);
    wire [1:0][1:0][31:0] A_frag_buf;
    wire [1:0][3:0][31:0] B_frag_buf;
    wire [1:0][3:0][31:0] C_frag_buf;

    wire valid_buf;
    wire ready_buf;

    wire enq = valid_in && ready_in;
    wire deq = valid_buf && ready_buf;
    wire empty;
    wire full;
    assign ready_in  = !full;
    assign valid_buf = !empty;

    // 'Issue queue' for the FEDP units.
    // This exists to decouple the execution of the dot-product unit from
    // the operand arrival.  Operands from execute_if can arrive
    // intermittently according to the frontend's behavior, and since the dpu
    // can also stall for a fixed initiation latency, we need to decouple the
    // two to efficiently feed the dpu.
    //
    // TODO: better queue design possible; e.g. B_frag is shared by two
    // threadgroups, so we need only 1 queue per octet for B
    VX_fifo_queue #(
        .DATAW ($bits(A_frag) + $bits(B_frag) + $bits(C_frag)),
        .DEPTH (ISSUE_QUEUE_DEPTH)
    ) input_buffer (
        .clk       (clk),
        .reset     (reset),
        .push      (enq),
        .pop       (deq),
        .data_in   ({A_frag,     B_frag,     C_frag}),
        .data_out  ({A_frag_buf, B_frag_buf, C_frag_buf}),
        .empty     (empty),
        `UNUSED_PIN(alm_empty),
        .full      (full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    wire [3:0] fedp_valids;
    wire fedp_valid_out = &(fedp_valids);
    wire fedp_ready_out = !stall;
    wire fedp_fire_out  = fedp_valid_out && fedp_ready_out;

    wire fedp_valid_in = valid_buf;
    wire fedp_ready_in = fedp_ready_out; // coupled
    wire fedp_fire_in  = fedp_valid_in && fedp_ready_in;

    // 0: FEDP uses first half from input_buffer
    // 1: FEDP uses last half and pops input_buffer
    wire step_in;
    // 0: FEDP produces first half of D_frag
    // 1: FEDP produces last half of D_frag and asserts valid_out
    wire step_out;
    assign ready_buf = fedp_fire_in && (step_in == 1'b1);

    wire  [3:0][31:0] D_reg;
    logic [3:0][31:0] D_reg_n;

    // Staging buffer that latches the D half-tile.
    VX_tensor_reg #(
        .N($bits(D_reg))
    ) staging_d (
        .clk(clk),
        .reset(reset),
        .d(D_reg_n),
        .en(1'b1),
        .q(D_reg)
    );

    // latch the first-half result of D_frag
    wire  [3:0][31:0] D_half;
    always @(*) begin
        D_reg_n = D_reg;
        if (fedp_fire_out) begin
            if (step_out == 1'b0) begin
                D_reg_n = D_half;
            end
        end
    end

    // flip step_in/step_out on FEDP in/out fire, respectively
    VX_tensor_reg #(
        .N(1)
    ) staging_step_in (
        .clk(clk),
        .reset(reset),
        .d(~step_in),
        .en(fedp_fire_in),
        .q(step_in)
    );
    VX_tensor_reg #(
        .N(1)
    ) staging_step_out (
        .clk(clk),
        .reset(reset),
        .d(~step_out),
        .en(fedp_fire_out),
        .q(step_out)
    );

    // TODO: Instead of latching half-result and constructing a full D tile,
    // we should be able to send these half fragments down to commit stage
    // immediately, saving flop space
    assign D_frag[0][0] = D_reg[0];
    assign D_frag[0][2] = D_reg[1];
    assign D_frag[1][0] = D_reg[2];
    assign D_frag[1][2] = D_reg[3];
    assign D_frag[0][1] = D_half[0];
    assign D_frag[0][3] = D_half[1];
    assign D_frag[1][1] = D_half[2];
    assign D_frag[1][3] = D_half[3];

    // 4 FEDPs per threadgroup
    for (genvar i = 0; i < 4; ++i) begin
        localparam int d_row = i / 2;
        localparam int d_col = (i % 2) * 2;
        // Dot product (FEDP) unit generated from Chisel
        TensorDotProductUnit fedp (
          .clock (clk),
          .reset (reset),
          .io_in_valid      (fedp_fire_in),
          .io_in_bits_a_0   (A_frag_buf[d_row][0]),
          .io_in_bits_a_1   (A_frag_buf[d_row][1]),
          .io_in_bits_a_2   (32'h0),
          .io_in_bits_a_3   (32'h0),
          .io_in_bits_b_0   (step_in == 1'b0 ? B_frag_buf[0][d_col] : B_frag_buf[0][d_col + 1]),
          .io_in_bits_b_1   (step_in == 1'b0 ? B_frag_buf[1][d_col] : B_frag_buf[1][d_col + 1]),
          .io_in_bits_b_2   (32'h0),
          .io_in_bits_b_3   (32'h0),
          .io_in_bits_c     (step_in == 1'b0 ? C_frag_buf[d_row][d_col] : C_frag_buf[d_row][d_col + 1]),
          .io_stall         (stall),
          .io_out_valid     (fedp_valids[i]),
          .io_out_bits_data (D_half[i])
        );
    end

    assign valid_out = fedp_valid_out && (step_out == 1'b1);
endmodule

`endif
