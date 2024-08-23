`ifdef EXT_T_ENABLE
`include "VX_fpu_define.vh"

// Module that contains the threadgroups with DPUs + operand buffer.
module VX_tensor_threadgroups #(
    parameter ISW,
    parameter OCTET,
    // @perf: has big impact on throughput.  A rule of thumb is to set it to
    // the pipeline length of FEDPs in order to make sure there are enough
    // entries to fully saturate the pipeline, but this is still rough
    parameter OPERAND_BUFFER_DEPTH = `LATENCY_HMMA
) (
    input clk,
    input reset,

    input valid_in,
    output ready_in,
    // [rows][cols][dtype]
    // (m,n,k) = (4,4,2)
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

    wire [3:0][1:0][31:0] A_tile_buf;
    wire [1:0][3:0][31:0] B_tile_buf;
    wire [3:0][3:0][31:0] C_tile_buf;

    wire wid_empty;
    wire wid_full;

    wire empty;
    wire full;
    // sync operand buffer and wid buffer
    assign ready_in = !full && !wid_full;

    wire [1:0] threadgroup_valids_out;
    wire [1:0] threadgroup_readys_in;
    wire threadgroup_valid_in = !empty;
    wire threadgroup_fire_in  = threadgroup_valid_in && &(threadgroup_readys_in);

    wire enq = valid_in && ready_in;
    wire deq = threadgroup_fire_in;

    // Operand buffer for the dot product units.
    //
    // This exists to decouple the execution of the dot-product unit from
    // the operand arrival.  Operands from the upstream execute_if can arrive
    // intermittently depending on the frontend's behavior, whereas downstream
    // writeback happens at a regular cadence.  Therefore to achieve full
    // throughput of the dpu, we need to decouple the operand arrival from the
    // direct input to the dpu.
    VX_fifo_queue #(
        .DATAW ($bits(A_tile) + $bits(B_tile) + $bits(C_tile)),
        .DEPTH (OPERAND_BUFFER_DEPTH)
    ) operand_buffer (
        .clk       (clk),
        .reset     (reset),
        .push      (enq),
        .pop       (deq),
        .data_in   ({A_tile,     B_tile,     C_tile}),
        .data_out  ({A_tile_buf, B_tile_buf, C_tile_buf}),
        .empty     (empty),
        `UNUSED_PIN(alm_empty),
        .full      (full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    // Split A_tile and C_tile by rows (0-1, 2-3) and parallelize in two
    // threadgroup DPUs; B_tile is shared across the two threadgroups. See
    // Figure 13 in paper
    VX_tensor_threadgroup #(
    ) threadgroup_0 (
        .clk   (clk),
        .reset (reset),
        .valid_in  (threadgroup_valid_in),
        .ready_in  (threadgroup_readys_in[0]),
        .stall     (!ready_out),
        .A_frag    (A_tile_buf[1:0]),
        .B_frag    (B_tile_buf),
        .C_frag    (C_tile_buf[1:0]),
        .valid_out (threadgroup_valids_out[0]),
        .D_frag    (D_tile[1:0])
    );
    VX_tensor_threadgroup #(
    ) threadgroup_1 (
        .clk   (clk),
        .reset (reset),
        .valid_in  (threadgroup_valid_in),
        .ready_in  (threadgroup_readys_in[1]),
        .stall     (!ready_out),
        .A_frag    (A_tile_buf[3:2]),
        .B_frag    (B_tile_buf),
        .C_frag    (C_tile_buf[3:2]),
        .valid_out (threadgroup_valids_out[1]),
        .D_frag    (D_tile[3:2])
    );

    `RUNTIME_ASSERT(&(threadgroup_valids_out) == |(threadgroup_valids_out),
                    ("threadgroups went out of sync!"))
    `RUNTIME_ASSERT(&(threadgroup_readys_in) == |(threadgroup_readys_in),
                    ("threadgroups went out of sync!"))

    wire wid_enq = valid_in && ready_in;
    wire wid_deq = valid_out && ready_out;

    assign valid_out = &(threadgroup_valids_out);

    // need to pass along warp id's to do multithreading
    VX_fifo_queue #(
        .DATAW   ($bits(wid)),
        // @perf: seems to require deeper depth than the FEDP issue queues to
        // not cause stalls.
        .DEPTH   (2 * OPERAND_BUFFER_DEPTH)
    ) wid_queue (
        .clk   (clk),
        .reset (reset),
        .push      (wid_enq),
        .pop       (wid_deq),
        .data_in   (wid),
        .data_out  (D_wid),
        .empty     (wid_empty),
        `UNUSED_PIN(alm_empty),
        .full      (wid_full),
        `UNUSED_PIN(alm_full),
        `UNUSED_PIN(size)
    );

    `RUNTIME_ASSERT(reset || !(deq && empty),
                    ("dequeueing from empty warp id queue!"))
endmodule

// does (m,n,k) = (2,4,2) matmul compute over 2 cycles.
// see Figure 10(b) of the paper.
module VX_tensor_threadgroup #(
    parameter HALF_PRECISION = 0
) (
    input clk,
    input reset,

    input valid_in,
    output ready_in,
    input stall,
    // all *_frag are row-major
    // (m,n,k) = (2,4,2)
    input [1:0][1:0][31:0] A_frag,
    input [1:0][3:0][31:0] B_frag,
    input [1:0][3:0][31:0] C_frag,

    output valid_out,
    output [1:0][3:0][31:0] D_frag
);
    wire fedp_valid_in;
    wire fedp_ready_in;
    wire fedp_fire_in  = fedp_valid_in && fedp_ready_in;

    wire [3:0] fedp_valids;
    wire fedp_valid_out = &(fedp_valids);
    wire fedp_ready_out = !stall;
    wire fedp_fire_out  = fedp_valid_out && fedp_ready_out;

    assign fedp_valid_in = valid_in;
    // coupled ready; backpressure immediately reaches input from output
    assign fedp_ready_in = fedp_ready_out;

    // The dot product units take 2 cycles to finish computing A_frag * B_frag
    // + C_frag.  substep_in and substep_out keeps track of which cycle they're at
    // & when they have to pop from input queue and push to result queue.
    // Note that substep is different from the "step" defined in the HMMA
    // instruction set; it is similar in meaning to the substeps in
    // VX_tensor_octet.
    //
    // substep_in == 0: FEDP uses first half from operand buffer
    // substep_in == 1: FEDP uses last half and pops from operand buffer
    wire substep_in;
    // substep_out == 0: FEDP produces first half of D_frag
    // substep_out == 1: FEDP produces last half of D_frag and asserts valid_out
    wire substep_out;
    assign ready_in = fedp_fire_in && (substep_in == 1'b1);

    wire  [3:0][31:0] D_reg;
    logic [3:0][31:0] D_reg_n;

    // staging buffer that latches the D half-tile
    VX_tensor_reg #(
        .DATAW($bits(D_reg))
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
            if (substep_out == 1'b0) begin
                D_reg_n = D_half;
            end
        end
    end

    // flip substep_in/substep_out on FEDP in/out fire, respectively
    VX_tensor_reg #(
        .DATAW(1)
    ) staging_substep_in (
        .clk(clk),
        .reset(reset),
        .d(~substep_in),
        .en(fedp_fire_in),
        .q(substep_in)
    );
    VX_tensor_reg #(
        .DATAW(1)
    ) staging_substep_out (
        .clk(clk),
        .reset(reset),
        .d(~substep_out),
        .en(fedp_fire_out),
        .q(substep_out)
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

    wire [1:0][3:0][15:0] A_frag_fp16 = A_frag;
    wire [3:0][3:0][15:0] B_frag_fp16 = B_frag;

    // 4 FEDPs per threadgroup
    for (genvar i = 0; i < 4; ++i) begin
        // Determine which elements in the D matrix the dot-product units get
        // mapped to.
        //
        // At substep == 0, the 0th and 2nd columns of D begins compute;
        // At substep == 1, the 1st and 3rd columns of D begins compute.
        // There are two row elements for each column, rounding out to
        // 4 elements computed by 4 FEDPs at every cycle
        // (see Figure 10(b)).

        //     i    : 0, 1, 2, 3
        // d_row    : 0, 0, 1, 1
        // d_col    : 0, 2, 0, 2
        // d_col_sel: 1, 3, 1, 3
        //
        // substep 0:
        // [ 0 x 2 x ]
        // [ 1 x 3 x ]
        // substep 1:
        // [ x 0 x 2 ]
        // [ x 1 x 3 ]
        localparam int d_row = i / 2;
        localparam int d_col = (i % 2) * 2;
        wire [31:0] d_col_sel = (substep_in == 1'b0) ? d_col : (d_col + 1);

        // Dot product (FEDP) unit generated from Chisel
        if (HALF_PRECISION != 0) begin
            TensorDotProductUnit fedp (
              .clock (clk),
              .reset (reset),
              .io_in_valid      (fedp_fire_in),
              .io_in_bits_a_0   (A_frag[d_row][0][15: 0]),
              .io_in_bits_a_1   (A_frag[d_row][0][31:16]),
              .io_in_bits_a_2   (A_frag[d_row][1][15: 0]),
              .io_in_bits_a_3   (A_frag[d_row][1][31:16]),
              .io_in_bits_b_0   (B_frag[0][d_col_sel][15: 0]),
              .io_in_bits_b_1   (B_frag[0][d_col_sel][31:16]),
              .io_in_bits_b_2   (B_frag[1][d_col_sel][15: 0]),
              .io_in_bits_b_3   (B_frag[1][d_col_sel][31:16]),
              .io_in_bits_c     (C_frag[d_row][d_col_sel]),
              .io_stall         (stall),
              .io_out_valid     (fedp_valids[i]),
              .io_out_bits_data (D_half[i])
            );
        end else begin
            TensorDotProductUnit fedp (
              .clock (clk),
              .reset (reset),
              .io_in_valid      (fedp_fire_in),
              .io_in_bits_a_0   (A_frag[d_row][0]),
              .io_in_bits_a_1   (A_frag[d_row][1]),
              .io_in_bits_a_2   (32'h0),
              .io_in_bits_a_3   (32'h0),
              .io_in_bits_b_0   (B_frag[0][d_col_sel]),
              .io_in_bits_b_1   (B_frag[1][d_col_sel]),
              .io_in_bits_b_2   (32'h0),
              .io_in_bits_b_3   (32'h0),
              .io_in_bits_c     (C_frag[d_row][d_col_sel]),
              .io_stall         (stall),
              .io_out_valid     (fedp_valids[i]),
              .io_out_bits_data (D_half[i])
            );
        end
    end

    assign valid_out = fedp_valid_out && (substep_out == 1'b1);
endmodule

`endif
