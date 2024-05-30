`ifdef EXT_T_ENABLE
`include "VX_fpu_define.vh"

module VX_tensor_dpu #(
    parameter ISW,
    parameter OCTET
) (
    input clk,
    input reset,

    input valid_in,
    output ready_in,
    input [3:0][1:0][31:0] A_tile,
    input [1:0][3:0][31:0] B_tile,
    input [3:0][3:0][31:0] C_tile,
    input [`NW_WIDTH-1:0]  wid,

    output valid_out,
    input  ready_out,
    output [3:0][3:0][31:0] D_tile,
    output [`NW_WIDTH-1:0]  D_wid
);
    logic [3:0][3:0][31:0] result_hmma;

    always @(*) begin
        dpi_hmma(valid_in, A_tile, B_tile, C_tile, result_hmma);
    end

    logic ready_reg;
    always @(posedge clk) begin
        if (reset) begin
            ready_reg <= '1;
        end else if (valid_in && ready_in) begin
            ready_reg <= '0;
            dpi_print_results(int'(ISW), int'(OCTET), A_tile, B_tile, C_tile, result_hmma);
        end else if (valid_out && ready_out) begin
            ready_reg <= '1;
        end
    end

    // ready as soon as valid_out
    // assign ready_in = ready_reg;

    // fully pipelined; ready_in is coupled to ready_out by immediately
    // stalling
    // assign ready_in = ready_out;

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

    logic [1:0] threadgroup_valids;
    logic [1:0] threadgroup_readys;
    // B_tile is shared across the two threadgroups; see Figure 13
    VX_tensor_threadgroup #(
    ) threadgroup_0 (
        .clk   (clk),
        .reset (reset),
        .valid_in  (valid_in),
        .ready_in  (threadgroup_readys[0]),
        .stall     (!ready_out),
        .A_frag    (A_tile[1:0]),
        .B_frag    (B_tile),
        .C_frag    (C_tile[1:0]),
        .valid_out (threadgroup_valids[0]),
        .D_frag    (D_tile[1:0])
    );
    VX_tensor_threadgroup #(
    ) threadgroup_1 (
        .clk   (clk),
        .reset (reset),
        .valid_in  (valid_in),
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

    assign ready_in  = &(threadgroup_readys);
    assign valid_out = &(threadgroup_valids);

    // need to pass along warp id's to do multithreading
    VX_fifo_queue #(
        .DATAW   ($bits(wid)),
        .DEPTH   (`LATENCY_HMMA + `LATENCY_HMMA)
    ) wid_queue (
        .clk   (clk),
        .reset (reset),
        .push      (enq),
        .pop       (deq),
        .data_in   (wid),
        .data_out  (D_wid),
        .empty     (empty),
        `UNUSED_PIN(alm_empty),
        .full      (full), // should be impossible to overflow
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    `RUNTIME_ASSERT(reset || !full, ("dpu wid queue is full!"))

    // `RUNTIME_ASSERT(reset || (&(threadgroup_valids) == valid_out),
    //                 ("FEDP and metadata queue went out of sync!"))
endmodule

// does (m,n,k) = (2,4,2) matmul compute over 2 cycles.
// matches Figure 10(b) of the paper.
module VX_tensor_threadgroup #(
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

    VX_fifo_queue #(
        .DATAW ($bits(A_frag) + $bits(B_frag) + $bits(C_frag)),
        .DEPTH   (4)
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

    logic [3:0] fedp_valids;
    wire fedp_valid_out = &(fedp_valids);
    wire fedp_ready_out = !stall;
    wire fedp_fire_out  = fedp_valid_out && fedp_ready_out;

    wire fedp_valid_in = valid_buf;
    wire fedp_ready_in = fedp_ready_out; // coupled
    wire fedp_fire_in  = fedp_valid_in && fedp_ready_in;

    // 0: FEDP uses first half from input_buffer
    // 1: FEDP uses last half and pops input_buffer
    logic step_in;
    // 0: FEDP produces first half of D_frag
    // 1: FEDP produces last half of D_frag and asserts valid_out
    logic step_out;
    assign ready_buf = fedp_fire_in && (step_in == 1'b1);

    // FIXME shrink size
    logic [1:0][3:0][31:0] D_reg, D_reg_n;
    wire  [3:0][31:0] D_half;
    always @(*) begin
        D_reg_n = D_reg;

        if (fedp_fire_out) begin
            if (step_out == 1'b0) begin
                D_reg_n[0][0] = D_half[0];
                D_reg_n[0][2] = D_half[1];
                D_reg_n[1][0] = D_half[2];
                D_reg_n[1][2] = D_half[3];
            end
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            step_in <= '0;
            step_out <= '0;

            D_reg <= '0;
        end else begin
            if (fedp_fire_in) begin
                step_in <= ~step_in;
            end
            if (fedp_fire_out) begin
                step_out <= ~step_out;
            end

            D_reg <= D_reg_n;
        end
    end

    assign D_frag[0][0] = D_reg[0][0];
    assign D_frag[0][2] = D_reg[0][2];
    assign D_frag[1][0] = D_reg[1][0];
    assign D_frag[1][2] = D_reg[1][2];
    assign D_frag[0][1] = D_half[0];
    assign D_frag[0][3] = D_half[1];
    assign D_frag[1][1] = D_half[2];
    assign D_frag[1][3] = D_half[3];

    // 4 FEDPs per threadgroup
    for (genvar i = 0; i < 4; ++i) begin
        localparam int d_row = i / 2;
        localparam int d_col = (i % 2) * 2;
        // four-element dot product (FEDP) unit
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
