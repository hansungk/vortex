`ifdef EXT_T_ENABLE
`include "VX_fpu_define.vh"

module VX_tensor_dpu #(
    parameter ISW,
    parameter OCTET
) (
    input clk,
    input reset,

    input stall,

    input valid_in,
    output ready_in,
    input [3:0][1:0][31:0] A_tile,
    input [1:0][3:0][31:0] B_tile,
    input [3:0][3:0][31:0] C_tile,
    input [`NW_WIDTH-1:0]  wid,

    output valid_out,
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
        end else if (valid_in) begin
            ready_reg <= '0;
            dpi_print_results(int'(ISW), int'(OCTET), A_tile, B_tile, C_tile, result_hmma);
        end else if (valid_out) begin
            ready_reg <= '1;
        end
    end

    // ready as soon as valid_out
    assign ready_in = ready_reg || valid_out;

    // fully pipelined; always ready
    // assign ready_in = 1'b1;

    // wire        dpu_valid;
    // wire [31:0] dpu_data;
    // TensorDotProductUnit dpu_pipe (
    //   .clock (clk),
    //   .reset (reset),
    //   .io_in_valid  (valid_in && ready_in),
    //   .io_in_bits_a_0 (32'h40000000),
    //   .io_in_bits_a_1 (32'h40000000),
    //   .io_in_bits_a_2 (32'h40000000),
    //   .io_in_bits_a_3 (32'h40000000),
    //   .io_in_bits_b_0 (32'h40000000),
    //   .io_in_bits_b_1 (32'h40000000),
    //   .io_in_bits_b_2 (32'h40000000),
    //   .io_in_bits_b_3 (32'h40000000),
    //   .io_in_bits_c   (32'h3f800000),
    //   .io_out_valid (dpu_valid),
    //   .io_out_bits_data (dpu_data)
    // );

    logic [1:0] threadgroup_valids;
    // B_tile is shared across the two threadgroups; see Figure 13
    VX_tensor_threadgroup #(
    ) threadgroup_0 (
        .clk   (clk),
        .reset (reset),
        .valid_in  (valid_in && ready_in),
        .stall     (stall),
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
        .valid_in  (valid_in && ready_in),
        .stall     (stall),
        .A_frag    (A_tile[3:2]),
        .B_frag    (B_tile),
        .C_frag    (C_tile[3:2]),
        .valid_out (threadgroup_valids[1]),
        .D_frag    (D_tile[3:2])
    );

    // fixed-latency queue
    VX_shift_register #(
        .DATAW  (1 + $bits(wid)/* + $bits(D_tile)*/),
        // .DEPTH  (`LATENCY_HMMA),
        .DEPTH  (2),
        .RESETW (1)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in && ready_in, wid  /*, result_hmma*/}),
        .data_out ({valid_out,            D_wid/*, D_tile     */})
    );

    // FIXME: breaks when stall is on!
    `RUNTIME_ASSERT(reset || (&(threadgroup_valids) == valid_out),
                    ("FEDP and metadata queue went out of sync!"))
endmodule

// does (m,n,k) = (2,4,2) matmul compute over 2 cycles.
// matches Figure 10(b) of the paper.
module VX_tensor_threadgroup #(
) (
    input clk,
    input reset,

    input valid_in,
    input stall,
    input [1:0][1:0][31:0] A_frag,
    input [1:0][3:0][31:0] B_frag,
    input [1:0][3:0][31:0] C_frag,

    output valid_out,
    output [1:0][3:0][31:0] D_frag

);
    // 4 FEDPs per threadgroup
    // FIXME: experimenting with 8 FEDPs first
    logic [1:0][3:0] valids;
    for (genvar D_row = 0; D_row < 2; ++D_row) begin
      for (genvar D_col = 0; D_col < 4; ++D_col) begin
        // four-element dot product (FEDP) unit
        TensorDotProductUnit fedp (
          .clock (clk),
          .reset (reset),
          .io_in_valid      (valid_in),
          .io_in_bits_a_0   (A_frag[D_row][0]),
          .io_in_bits_a_1   (A_frag[D_row][1]),
          .io_in_bits_a_2   (32'h0),
          .io_in_bits_a_3   (32'h0),
          .io_in_bits_b_0   (B_frag[0][D_col]),
          .io_in_bits_b_1   (B_frag[1][D_col]),
          .io_in_bits_b_2   (32'h0),
          .io_in_bits_b_3   (32'h0),
          .io_in_bits_c     (C_frag[D_row][D_col]),
          .io_stall         (1'b0), // FIXME
          .io_out_valid     (valids[D_row][D_col]),
          .io_out_bits_data (D_frag[D_row][D_col])
        );
      end
    end

    assign valid_out = (&(valids[0])) && (&(valids[1]));

    `RUNTIME_ASSERT(reset || !stall, ("stall not supported yet in tensor dpu!"))
endmodule

`endif
