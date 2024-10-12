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
    localparam NUM_LANES = `NUM_THREADS;
    localparam METADATA_QUEUE_DEPTH = 2; // FIXME: arbitrary

    /* commit_if.data_t parts that we need to keep around:
        - uuid
        - wid
        - tmask
        - PC
        - wb
        - rd
    */
    wire [`NUM_WARPS-1:0][`UUID_WIDTH-1:0] execute_if_data_uuid;
    wire [`NUM_WARPS-1:0][`NW_WIDTH-1:0]   execute_if_data_wid;
    wire [`NUM_WARPS-1:0][NUM_LANES-1:0]   execute_if_data_tmask;
    wire [`NUM_WARPS-1:0][`XLEN-1:0]       execute_if_data_PC;
    wire [`NUM_WARPS-1:0]                  execute_if_data_wb;
    wire [`NUM_WARPS-1:0][`NR_BITS-1:0]    execute_if_data_rd;

    wire [`NUM_WARPS-1:0] metadata_queue_fulls;
    wire [`NUM_WARPS-1:0] metadata_queue_emptys;
    // OR not AND; we don't want any warp to be full
    wire metadata_queue_full = |(metadata_queue_fulls);
    assign execute_if.ready = !metadata_queue_full;

    `RUNTIME_ASSERT((!execute_if.valid || execute_if.data.wid == `NW_WIDTH'(0)),
        ("runtime error: WGMMA execute not supported for warps other than 0!"))

    wire metadata_deq;

    for (genvar i = 0; i < `NUM_WARPS; i++) begin
        // Metadata queue for commit_if.  This simply copies execute_if's
        // metadata and pops them in conjunction with commit fire.
        //
        // This has to be separated per-warp, as otherwise requests from
        // multiple warps can be enqueued interleaved, which makes it hard to
        // ensure two consecutive dequeues are associated with the same warp for
        // commit. (FIXME: this is not strictly necessary though.)

        wire operand_enq_fire = execute_if.valid && execute_if.ready;
        wire enq = operand_enq_fire && (execute_if.data.wid == `NW_WIDTH'(i));
        // FIXME: commit only warp 0
        wire deq = metadata_deq && (`NW_WIDTH'(i) == `NW_WIDTH'(0));

        localparam DATAW = `UUID_WIDTH + `NW_WIDTH + `NUM_THREADS + `XLEN + 1 + `NR_BITS;
        VX_fifo_queue #(
            .DATAW(DATAW),
            .DEPTH(METADATA_QUEUE_DEPTH)
        ) pending_uops (
            .clk(clk),
            .reset(reset),
            .push(enq),
            .pop(deq),
            .data_in({execute_if.data.uuid,  execute_if.data.wid,
                      execute_if.data.tmask, execute_if.data.PC,
                      execute_if.data.wb,    execute_if.data.rd}),
            .data_out({execute_if_data_uuid[i],  execute_if_data_wid[i],
                       execute_if_data_tmask[i], execute_if_data_PC[i],
                       execute_if_data_wb[i],    execute_if_data_rd[i]}),
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

    wire initiate_ready; // FIXME: unused
    wire writeback_valid;
    wire writeback_last;

    wire metadata_valid = ~metadata_queue_emptys[0/*FIXME*/];
    // dequeue metadata at the last writeback
    assign metadata_deq = metadata_valid && writeback_valid && writeback_last;

    VX_tensor_hopper_core #(
    ) tensor_hopper_core (
        .clk(clk),
        .reset(reset),

        .initiate_valid(metadata_valid),
        .initiate_wid(`NW_WIDTH'(0)/*FIXME*/),
        .initiate_ready(initiate_ready),

        .writeback_valid(writeback_valid),
        `UNUSED_PIN(writeback_wid),
        .writeback_last(writeback_last),
        .writeback_ready(commit_if.ready)
    );

    wire [`NUM_THREADS-1:0][`XLEN-1:0] wb_data = '0;

    assign commit_if.valid = writeback_valid;
    assign commit_if.data.uuid   = execute_if_data_uuid[0];
    assign commit_if.data.wid    = execute_if_data_wid[0];
    assign commit_if.data.tmask  = execute_if_data_tmask[0];
    assign commit_if.data.PC     = execute_if_data_PC[0];
    assign commit_if.data.wb     = writeback_last;
    // custom rd
    assign commit_if.data.rd     = (`NR_BITS'(`NUM_IREGS) + `NR_BITS'(4'd3/*FIXME*/));
    assign commit_if.data.data   = wb_data;
    assign commit_if.data.tensor = writeback_last;
    assign commit_if.data.pid    = 1'b0;
    assign commit_if.data.sop    = 1'b1;
    // eop is deliberately set so that we don't underflow the pending_instr
    // buffer in VX_schedule.  An instruction is considered committed only
    // when the eop bit is set to one (see VX_commit).
    assign commit_if.data.eop    = writeback_last;
endmodule


// TODO: replace this with a Chisel module
module VX_tensor_hopper_core #(
) (
    input clk,
    input reset,

    input                 initiate_valid,
    input [`NW_WIDTH-1:0] initiate_wid,
    output                initiate_ready,

    output                 writeback_valid,
    output [`NW_WIDTH-1:0] writeback_wid,
    // indicates if this is the last writeback for the given wid, in which
    // case the original HGMMA instruction should be signalled retired
    output                 writeback_last,
    input                  writeback_ready
);
    // dummy FSM that generates commits
    localparam STATE_IDLE = 4'd0;
    localparam STATE_FINISH = 4'd15;
    logic [3:0] state, state_n;

    assign initiate_ready = (state == STATE_IDLE);

    always @(*) begin
        state_n = state;

        case (state)
            STATE_IDLE: begin
                state_n = state;
            end
            STATE_FINISH: begin
                // hold until writeback_ready
                if (writeback_ready) begin
                    state_n = STATE_IDLE;
                end
            end
            default: begin
                state_n = state + 4'd1;
            end
        endcase

        // kick-off
        if (initiate_valid && initiate_ready) begin
            state_n = 4'd1;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= '0;
        end else begin
            state <= state_n;
        end
    end

    assign writeback_valid = (state != STATE_IDLE);
    assign writeback_wid = '0; // TODO
    assign writeback_last = (state == STATE_FINISH);

endmodule

`endif
