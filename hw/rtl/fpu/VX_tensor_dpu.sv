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

    output valid_out,
    output [3:0][3:0][31:0] D_tile
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

    VX_shift_register #(
        .DATAW  (1 + $bits(D_tile)),
        .DEPTH  (`LATENCY_HMMA),
        .RESETW (1)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall),
        .data_in  ({valid_in, result_hmma}),
        .data_out ({valid_out, D_tile})
    );
endmodule
`endif
