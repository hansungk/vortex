`include "VX_fpu_define.vh"

module VX_tensor_core #(

) (
    input clk,
    input reset,

    VX_dispatch_if.slave dispatch_if [`ISSUE_WIDTH],
    VX_commit_if.master commit_if [`ISSUE_WIDTH]
);
    `STATIC_ASSERT(`NUM_THREADS == 32, ("tensor core requires # of threads in a warp to be 32"));
    `UNUSED_VAR(clk);
    `UNUSED_VAR(reset);
endmodule
