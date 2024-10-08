`ifdef EXT_T_ENABLE
`include "VX_fpu_define.vh"

module VX_tensor_hopper_core_block import VX_gpu_pkg::*; #(
    parameter ISW,
    parameter FP16
) (
    input clk,
    input reset,

    VX_execute_if.slave execute_if,
    VX_commit_if.master commit_if
);
    localparam METADATA_QUEUE_DEPTH = 16; // FIXME: arbitrary

    /* commit_if.data_t parts that we need to keep around:
        - uuid
        - wid
        - tmask
        - PC
        - wb
        - rd
    */

    localparam DATAW = `UUID_WIDTH + `NW_WIDTH + `NUM_THREADS + `XLEN + 1 + `NR_BITS;

    wire operand_enq_fire = execute_if.valid && execute_if.ready;
    wire commit_if_fire = commit_if.valid && commit_if.ready;
    wire [DATAW-1:0] execute_if_data_enq = {
        execute_if.data.uuid,
        execute_if.data.wid,
        execute_if.data.tmask,
        execute_if.data.PC,
        execute_if.data.wb,
        execute_if.data.rd
        // pid/sop/eop set later
    };

    wire [`NUM_WARPS-1:0][DATAW-1:0] execute_if_data_deq;
    logic [DATAW-1:0] execute_if_data_new_rd;

    wire [`NUM_WARPS-1:0] metadata_queue_fulls;
    wire [`NUM_WARPS-1:0] metadata_queue_emptys;
    // OR not AND, we don't want any warp full
    wire metadata_queue_full = |(metadata_queue_fulls);
    assign execute_if.ready = !metadata_queue_full;

    `RUNTIME_ASSERT((!execute_if.valid || execute_if.data.wid == `NW_WIDTH'(0)),
        ("runtime error: WGMMA execute not supported for warps other than 0!"))

    logic metadata_deq;

    for (genvar i = 0; i < `NUM_WARPS; i++) begin
        // Metadata queue for commit_if.  This simply copies execute_if's
        // metadata and pops them in conjunction with commit fire.
        //
        // This has to be separated per-warp, as otherwise requests from
        // multiple warps can be enqueued interleaved, which makes it hard to
        // ensure two consecutive dequeues are associated with the same warp for
        // commit. (FIXME: this is not strictly necessary though.)

        wire enq = operand_enq_fire && (execute_if.data.wid == `NW_WIDTH'(i));
        // FIXME: commit only warp 0
        wire deq = metadata_deq && commit_if.ready && (`NW_WIDTH'(i) == `NW_WIDTH'(0));

        VX_fifo_queue #(
            .DATAW(DATAW),
            .DEPTH(METADATA_QUEUE_DEPTH)
        ) pending_uops (
            .clk(clk),
            .reset(reset),
            .push(enq),
            .pop(deq),
            .data_in(execute_if_data_enq),
            .data_out(execute_if_data_deq[i]),
            .empty(metadata_queue_emptys[i]),
            `UNUSED_PIN(alm_empty),
            .full(metadata_queue_fulls[i]),
            `UNUSED_PIN(alm_full),
            `UNUSED_PIN(size)
        );
    end

    // this shouldn't really happen unless there's a big contention over
    // the commit stage
    `RUNTIME_ASSERT(!(!reset && metadata_queue_full), ("tensor core uop queue is full!"))

    // dummy FSM that generates commits
    logic [1:0] state, state_n;
    localparam STATE_IDLE = 4'd0;

    always @(*) begin
        state_n = state;
        metadata_deq = 1'b0;

        // when incremented to 1, count up until wrap-around to 0
        if (state != STATE_IDLE) begin
            state_n = state + 1'd1;
        end else begin
            // kick-off from idle when execute valid
            // FIXME: only checks warp 0 for commit!
            if (~metadata_queue_emptys[0/*FIXME*/]) begin
                state_n = 4'd1;
            end
        end

        // dequeue metadata when wrapping around
        if ((state != STATE_IDLE) && (state_n == STATE_IDLE)) begin
            metadata_deq = 1'b1;
        end

        // change rd of the commit data according to state
        execute_if_data_new_rd =
            {execute_if_data_deq[0/*FIXME*/][DATAW-1:`NR_BITS],
             (`NR_BITS'(`NUM_IREGS) + `NR_BITS'(state))};
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= '0;
        end else begin
            state <= state_n;
        end
    end

    // assign commit_if.valid = metadata_deq;
    assign commit_if.valid = (state != STATE_IDLE);

    wire [`NUM_THREADS-1:0][`XLEN-1:0] wb_data = '0;

    localparam COMMIT_DATAW = `UUID_WIDTH + `NW_WIDTH + `NUM_THREADS + `XLEN + 1 + `NR_BITS + (`NUM_THREADS * `XLEN) + 1 + 1 + 1;
    wire [COMMIT_DATAW-1:0] commit_if_data = {
        // write-back to the correct rd only when eop
        ((state == 2'b11) ? execute_if_data_deq[0/*FIXME*/] : execute_if_data_new_rd), /* uuid ~ rd */
        wb_data, /* data */
        1'b0, /* pid */
        1'b1, /* sop */
        (state == 2'b11)  /* eop */
        // 1'b1  /* eop */
    };

    assign commit_if.data = commit_if_data;
endmodule

`endif
