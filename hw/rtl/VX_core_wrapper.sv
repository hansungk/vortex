`include "VX_define.vh"
`include "VX_gpu_pkg.sv"
// TODO: move VX_define constants to parameters, and then parameterize in blackbox

module Vortex import VX_gpu_pkg::*; #(
    parameter CORE_ID = 0,
    parameter BOOTROM_HANG100 = 32'h10100,
    parameter NUM_THREADS = 0
) (

    /* adapt to CoreIO bundle at src/main/scala/tile/Core.scala */

    input         clock,
    input         reset,
    // input         hartid,
    input  [31:0] reset_vector,
    input         interrupts_debug,
    input         interrupts_mtip,
    input         interrupts_msip,
    input         interrupts_meip,
    input         interrupts_seip,

    input         imem_0_a_ready,
    input         imem_0_d_valid,
    input  [2:0]  imem_0_d_bits_opcode,
    // input  [1:0]  imem_0_d_bits_param,
    input  [3:0]  imem_0_d_bits_size,
    input  [ICACHE_TAG_WIDTH-1:0]  imem_0_d_bits_source,
    // input  [2:0]  imem_0_d_bits_sink,
    // input         imem_0_d_bits_denied,
    input  [31:0] imem_0_d_bits_data,
    // input         imem_0_d_bits_corrupt,
    output        imem_0_a_valid,
    output [2:0]  imem_0_a_bits_opcode,
    // output [2:0]  imem_0_a_bits_param,
    output [3:0]  imem_0_a_bits_size,
    output [ICACHE_TAG_WIDTH-1:0]  imem_0_a_bits_source,
    output [31:0] imem_0_a_bits_address,
    output [3:0]  imem_0_a_bits_mask,
    output [31:0] imem_0_a_bits_data,
    // output        imem_0_a_bits_corrupt,
    output        imem_0_d_ready,

    input         dmem_0_a_ready,
    input         dmem_0_d_valid,
    input  [2:0]  dmem_0_d_bits_opcode,
    // input  [1:0]  dmem_0_d_bits_param,
    input  [3:0]  dmem_0_d_bits_size,
    input  [DCACHE_NOSM_TAG_WIDTH-1:0]  dmem_0_d_bits_source,
    // input  [2:0]  dmem_0_d_bits_sink,
    // input         dmem_0_d_bits_denied,
    input  [31:0] dmem_0_d_bits_data,
    // input         dmem_0_d_bits_corrupt,
    output        dmem_0_a_valid,
    output [2:0]  dmem_0_a_bits_opcode,
    // output [2:0]  dmem_0_a_bits_param,
    output [3:0]  dmem_0_a_bits_size,
    output [DCACHE_NOSM_TAG_WIDTH-1:0]  dmem_0_a_bits_source,
    output [31:0] dmem_0_a_bits_address,
    output [3:0]  dmem_0_a_bits_mask,
    output [31:0] dmem_0_a_bits_data,
    // output        dmem_0_a_bits_corrupt,
    output        dmem_0_d_ready,

    input         dmem_1_a_ready,
    input         dmem_1_d_valid,
    input  [2:0]  dmem_1_d_bits_opcode,
    // input  [1:0]  dmem_1_d_bits_param,
    input  [3:0]  dmem_1_d_bits_size,
    input  [DCACHE_NOSM_TAG_WIDTH-1:0]  dmem_1_d_bits_source,
    // input  [2:0]  dmem_1_d_bits_sink,
    // input         dmem_1_d_bits_denied,
    input  [31:0] dmem_1_d_bits_data,
    // input         dmem_1_d_bits_corrupt,
    output        dmem_1_a_valid,
    output [2:0]  dmem_1_a_bits_opcode,
    // output [2:0]  dmem_1_a_bits_param,
    output [3:0]  dmem_1_a_bits_size,
    output [DCACHE_NOSM_TAG_WIDTH-1:0]  dmem_1_a_bits_source,
    output [31:0] dmem_1_a_bits_address,
    output [3:0]  dmem_1_a_bits_mask,
    output [31:0] dmem_1_a_bits_data,
    // output        dmem_1_a_bits_corrupt,
    output        dmem_1_d_ready,

    input         dmem_2_a_ready,
    input         dmem_2_d_valid,
    input  [2:0]  dmem_2_d_bits_opcode,
    // input  [1:0]  dmem_2_d_bits_param,
    input  [3:0]  dmem_2_d_bits_size,
    input  [DCACHE_NOSM_TAG_WIDTH-1:0]  dmem_2_d_bits_source,
    // input  [2:0]  dmem_2_d_bits_sink,
    // input         dmem_2_d_bits_denied,
    input  [31:0] dmem_2_d_bits_data,
    // input         dmem_2_d_bits_corrupt,
    output        dmem_2_a_valid,
    output [2:0]  dmem_2_a_bits_opcode,
    // output [2:0]  dmem_2_a_bits_param,
    output [3:0]  dmem_2_a_bits_size,
    output [DCACHE_NOSM_TAG_WIDTH-1:0]  dmem_2_a_bits_source,
    output [31:0] dmem_2_a_bits_address,
    output [3:0]  dmem_2_a_bits_mask,
    output [31:0] dmem_2_a_bits_data,
    // output        dmem_2_a_bits_corrupt,
    output        dmem_2_d_ready,

    input         dmem_3_a_ready,
    input         dmem_3_d_valid,
    input  [2:0]  dmem_3_d_bits_opcode,
    // input  [1:0]  dmem_3_d_bits_param,
    input  [3:0]  dmem_3_d_bits_size,
    input  [DCACHE_NOSM_TAG_WIDTH-1:0]  dmem_3_d_bits_source,
    // input  [2:0]  dmem_3_d_bits_sink,
    // input         dmem_3_d_bits_denied,
    input  [31:0] dmem_3_d_bits_data,
    // input         dmem_3_d_bits_corrupt,
    output        dmem_3_a_valid,
    output [2:0]  dmem_3_a_bits_opcode,
    // output [2:0]  dmem_3_a_bits_param,
    output [3:0]  dmem_3_a_bits_size,
    output [DCACHE_NOSM_TAG_WIDTH-1:0]  dmem_3_a_bits_source,
    output [31:0] dmem_3_a_bits_address,
    output [3:0]  dmem_3_a_bits_mask,
    output [31:0] dmem_3_a_bits_data,
    // output        dmem_3_a_bits_corrupt,
    output        dmem_3_d_ready,

    // input         fpu_fcsr_flags_valid,
    // input  [4:0]  fpu_fcsr_flags_bits,
    // // input  [63:0] fpu_store_data,
    // input  [31:0] fpu_toint_data,
    // input         fpu_fcsr_rdy,
    // input         fpu_nack_mem,
    // input         fpu_illegal_rm,
    // input         fpu_dec_wen,
    // input         fpu_dec_ldst,
    // input         fpu_dec_ren1,
    // input         fpu_dec_ren2,
    // input         fpu_dec_ren3,
    // input         fpu_dec_swap12,
    // input         fpu_dec_swap23,
    // input  [1:0]  fpu_dec_typeTagIn,
    // input  [1:0]  fpu_dec_typeTagOut,
    // input         fpu_dec_fromint,
    // input         fpu_dec_toint,
    // input         fpu_dec_fastpipe,
    // input         fpu_dec_fma,
    // input         fpu_dec_div,
    // input         fpu_dec_sqrt,
    // input         fpu_dec_wflags,
    // input         fpu_sboard_set,
    // input         fpu_sboard_clr,
    // input  [4:0]  fpu_sboard_clra,

    // output        fpu_hartid,
    // output [31:0] fpu_time,
    // output [31:0] fpu_inst,
    // output [31:0] fpu_fromint_data,
    // output [2:0]  fpu_fcsr_rm,
    // output        fpu_dmem_resp_val,
    // output [2:0]  fpu_dmem_resp_type,
    // output [4:0]  fpu_dmem_resp_tag,
    // output        fpu_valid,
    // output        fpu_killx,
    // output        fpu_killm,
    // output        fpu_keep_clock_enabled,

    output        cease,

    input         traceStall,
    output        wfi
);

    logic [3:0] intr_counter;
    logic msip_1d, intr_reset;
    logic busy;

    assign intr_reset = |intr_counter;
    /* interrupts */
    always @(posedge clock) begin
        msip_1d <= interrupts_msip;
        if (reset) begin
            intr_counter <= 4'h0;
        end else if (~msip_1d && interrupts_msip) begin
            // rising edge
            intr_counter <= 4'h6;
        end else begin
            intr_counter <= intr_counter > 0 ? intr_counter - 4'h1 : 4'h0;
        end
    end

    // ------------------------------------------------------------------------
    // TL <-> Vortex core-cache interface adapter
    // ------------------------------------------------------------------------

    VX_mem_bus_if #(
        .DATA_SIZE (ICACHE_WORD_SIZE), 
        .TAG_WIDTH (ICACHE_TAG_WIDTH)
    ) icache_bus_if();

    // NOTE(hansung): need to use DCACHE_NOSM_TAG_WIDTH here instead of
    // DCACHE_TAG_WIDTH; the latter is only used inside the core to
    // differentiate between requests going to the outside cache vs. going to
    // the shared memory.
    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_NOSM_TAG_WIDTH)
    ) dcache_bus_if[DCACHE_NUM_REQS]();

    // always @(posedge clock) begin
    //   `ASSERT(DCACHE_NUM_REQS == NUM_THREADS, "DCACHE_NUM_REQS doesn't match NUM_THREADS");
    // end

    /* imem */
    assign icache_bus_if.rsp_valid = imem_0_d_valid;
    // TODO: hardcoded DCACHE_WORD_SIZE = 4
    assign icache_bus_if.rsp_data.data = imem_0_d_bits_data;
    assign icache_bus_if.rsp_data.tag = imem_0_d_bits_source[ICACHE_TAG_WIDTH-1:0];
    assign imem_0_d_ready = icache_bus_if.rsp_ready;

    // always @(posedge clock) begin
    //     if (icache_req_if.valid && icache_req_if.ready)
    //         icache_rsp_if.tag <= icache_req_if.tag;
    // end
    assign imem_0_a_bits_source = {32'b0, icache_bus_if.req_data.tag}[ICACHE_TAG_WIDTH-1:0];
    assign imem_0_a_valid = icache_bus_if.req_valid;
    assign imem_0_a_bits_address = {icache_bus_if.req_data.addr, 2'b0};
    assign icache_bus_if.req_ready = imem_0_a_ready;

    assign imem_0_a_bits_data = 32'd0;
    assign imem_0_a_bits_mask = 4'hf;
    // assign imem_0_a_bits_corrupt = 1'b0;
    // assign imem_0_a_bits_param = 3'd0;
    assign imem_0_a_bits_size = 4'd2; // 32b
    assign imem_0_a_bits_opcode = 3'd4; // Get

    /* dmem */
    // Vortex core does not accept write acks; filter them out here
    assign dcache_bus_if[0].rsp_valid = 
        (dmem_0_d_valid && (dmem_0_d_bits_opcode !== 3'd0 /*AccessAck*/));
    assign dcache_bus_if[1].rsp_valid = 
        (dmem_1_d_valid && (dmem_1_d_bits_opcode !== 3'd0 /*AccessAck*/));
    assign dcache_bus_if[2].rsp_valid = 
        (dmem_2_d_valid && (dmem_2_d_bits_opcode !== 3'd0 /*AccessAck*/));
    assign dcache_bus_if[3].rsp_valid = 
        (dmem_3_d_valid && (dmem_3_d_bits_opcode !== 3'd0 /*AccessAck*/));

    assign dcache_bus_if[0].rsp_data.data = dmem_0_d_bits_data;
    assign dcache_bus_if[1].rsp_data.data = dmem_1_d_bits_data;
    assign dcache_bus_if[2].rsp_data.data = dmem_2_d_bits_data;
    assign dcache_bus_if[3].rsp_data.data = dmem_3_d_bits_data;

    // get tag (source) from one of the valid dmem lanes; any is fine, use
    // priority logic for simplicity
    // logic [9:0] tag_d;
    // always @(*) begin
    //     tag_d = '0;
    //     for (integer i = 0; i < 4; i += 1) begin
    //         if ({dmem_3_d_valid, dmem_2_d_valid, dmem_1_d_valid, dmem_0_d_valid}[i]) begin
    //             tag_d = {dmem_3_d_bits_source, dmem_2_d_bits_source, dmem_1_d_bits_source, dmem_0_d_bits_source}[i * 10 +: 10];
    //         end
    //     end
    // end
    // assign dcache_rsp_if.tag = tag_d;

    assign dcache_bus_if[0].rsp_data.tag = dmem_0_d_bits_source;
    assign dcache_bus_if[1].rsp_data.tag = dmem_1_d_bits_source;
    assign dcache_bus_if[2].rsp_data.tag = dmem_2_d_bits_source;
    assign dcache_bus_if[3].rsp_data.tag = dmem_3_d_bits_source;

    // When there's a write ACK coming back, ready bit should always be 1 to
    // accept them because core does not accept them on their own
    assign dmem_0_d_ready = dcache_bus_if[0].rsp_ready ||
                            (dmem_0_d_valid && (dmem_0_d_bits_opcode == 3'd0 /*AccessAck*/));
    assign dmem_1_d_ready = dcache_bus_if[1].rsp_ready ||
                            (dmem_1_d_valid && (dmem_1_d_bits_opcode == 3'd0 /*AccessAck*/));
    assign dmem_2_d_ready = dcache_bus_if[2].rsp_ready ||
                            (dmem_2_d_valid && (dmem_2_d_bits_opcode == 3'd0 /*AccessAck*/));
    assign dmem_3_d_ready = dcache_bus_if[3].rsp_ready ||
                            (dmem_3_d_valid && (dmem_3_d_bits_opcode == 3'd0 /*AccessAck*/));

    assign dmem_0_a_valid = dcache_bus_if[0].req_valid;
    assign dmem_1_a_valid = dcache_bus_if[1].req_valid;
    assign dmem_2_a_valid = dcache_bus_if[2].req_valid;
    assign dmem_3_a_valid = dcache_bus_if[3].req_valid;

    assign dmem_0_a_bits_address = {dcache_bus_if[0].req_data.addr, 2'b0};
    assign dmem_1_a_bits_address = {dcache_bus_if[1].req_data.addr, 2'b0};
    assign dmem_2_a_bits_address = {dcache_bus_if[2].req_data.addr, 2'b0};
    assign dmem_3_a_bits_address = {dcache_bus_if[3].req_data.addr, 2'b0};

    assign dmem_0_a_bits_data = dcache_bus_if[0].req_data.data;
    assign dmem_1_a_bits_data = dcache_bus_if[1].req_data.data;
    assign dmem_2_a_bits_data = dcache_bus_if[2].req_data.data;
    assign dmem_3_a_bits_data = dcache_bus_if[3].req_data.data;

    assign dmem_0_a_bits_source = dcache_bus_if[0].req_data.tag;
    assign dmem_1_a_bits_source = dcache_bus_if[1].req_data.tag;
    assign dmem_2_a_bits_source = dcache_bus_if[2].req_data.tag;
    assign dmem_3_a_bits_source = dcache_bus_if[3].req_data.tag;

    // we assume all lanes always have the same tag; otherwise the sourceId
    // logic in the Chisel tile breaks
    // always @(*) begin
    //   for (i = 0; i < 4; i++) begin
    //     assert(dcache_req_if.tag[0] == dcache_req_if.tag[i])
    //   end
    // end

    // Translate Vortex rw/byteen to TileLink opcode
    assign dmem_0_a_bits_opcode = 
        dcache_bus_if[0].req_data.rw ?
        (&dcache_bus_if[0].req_data.byteen ? 3'd0 /*PutFull*/ : 3'd0 /*FIXME: PutPartial but spoofed to PutFull*/)
        : 3'd4 /*Get*/;
    assign dmem_1_a_bits_opcode = 
        dcache_bus_if[1].req_data.rw ?
        (&dcache_bus_if[1].req_data.byteen ? 3'd0 /*PutFull*/ : 3'd0 /*FIXME: PutPartial but spoofed to PutFull*/)
        : 3'd4 /*Get*/;
    assign dmem_2_a_bits_opcode = 
        dcache_bus_if[2].req_data.rw ?
        (&dcache_bus_if[2].req_data.byteen ? 3'd0 /*PutFull*/ : 3'd0 /*FIXME: PutPartial but spoofed to PutFull*/)
        : 3'd4 /*Get*/;
    assign dmem_3_a_bits_opcode = 
        dcache_bus_if[3].req_data.rw ?
        (&dcache_bus_if[3].req_data.byteen ? 3'd0 /*PutFull*/ : 3'd0 /*FIXME: PutPartial but spoofed to PutFull*/)
        : 3'd4 /*Get*/;

    // Vortex cache requests are single-fixed-size
    // NOTE: MAKE SURE TO CHANGE CONSTANT WIDTH FOR SIZE!
    assign dmem_0_a_bits_size = 4'd2;
    assign dmem_1_a_bits_size = 4'd2;
    assign dmem_2_a_bits_size = 4'd2;
    assign dmem_3_a_bits_size = 4'd2;
    /* $countones(dcache_req_if.byteen[0]) === 'd4 ? 2'd2 :
        ($countones(dcache_req_if.byteen[0]) === 'd2 ? 2'd1 : 2'd0); */

    // byteen can be directly used as TL mask
    assign dmem_0_a_bits_mask = dcache_bus_if[0].req_data.byteen;
    assign dmem_1_a_bits_mask = dcache_bus_if[1].req_data.byteen;
    assign dmem_2_a_bits_mask = dcache_bus_if[2].req_data.byteen;
    assign dmem_3_a_bits_mask = dcache_bus_if[3].req_data.byteen;

    assign dcache_bus_if[0].req_ready = dmem_0_a_ready;
    assign dcache_bus_if[1].req_ready = dmem_1_a_ready;
    assign dcache_bus_if[2].req_ready = dmem_2_a_ready;
    assign dcache_bus_if[3].req_ready = dmem_3_a_ready;

    /* fpu */

    // assign {fpu_hartid, fpu_time, fpu_inst, fpu_fromint_data, fpu_fcsr_rm, fpu_dmem_resp_val, fpu_dmem_resp_type,
    //         fpu_dmem_resp_tag, fpu_valid, fpu_killx, fpu_killm, fpu_keep_clock_enabled} = '0;

    assign cease = ~busy;
    assign wfi = 1'b0; // FIXME: unused

    genvar i;
    generate for (i = 0; i < 4; i++) begin
        always @(posedge clock) begin
            if (dcache_bus_if[i].req_valid && dcache_bus_if[i].req_ready && dcache_bus_if[i].req_data.rw) begin
                // anything that starts with 0xC is heap address
                if ({dcache_bus_if[i].req_data.addr, 2'b0}[31:28] == 4'hc) begin
                    $display("[%d] STORE HEAP MEM: CORE=%d, THREAD=%d, ADDRESS=0x%X, DATA=0x%08X",
                             $time(), CORE_ID, i, {dcache_bus_if[i].req_data.addr, 2'b0}, dcache_bus_if[i].req_data.data);
                end
            end
        end
    end
    endgenerate

    logic sim_ebreak;
    logic [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value;

    logic [3:0] reset_start_counter;
    logic core_reset;
    logic dcr_reset;

    always @(posedge clock) begin
      if (reset) begin
        reset_start_counter <= 4'ha;
      end else begin
        if (reset_start_counter > 4'h0) begin
          reset_start_counter <= reset_start_counter - 4'h1;
        end
      end
    end
    // Delay reset signal by a few cycles to make time for resetting the DCR
    // (device configuration registers).
    assign core_reset = reset || (reset_start_counter != 4'h0); // || intr_reset;
    assign dcr_reset = !reset && (reset_start_counter != 4'h0);

    // A small FSM that tries to set DCR "properly" in the same order as
    // defined in VX_types.vh.
    //
    // DCR is a device configuration register that holds (among other things)
    // the startup address for the kernel, nominally set to 0x80000000.
    // TODO: Original Vortex code buffers dcr_bus by one cycle when
    // SOCKET_SIZE > 1, as below.  Might want to check if we need to do the
    // same
    //   `BUFFER_DCR_BUS_IF (core_dcr_bus_if, dcr_bus_if, (`SOCKET_SIZE > 1));
    logic [`VX_DCR_ADDR_BITS-1:0]  dcr_state;
    logic [`VX_DCR_ADDR_BITS-1:0]  dcr_state_n;
    logic                          dcr_write_valid;
    logic [`VX_DCR_ADDR_WIDTH-1:0] dcr_write_addr;
    logic [`VX_DCR_DATA_WIDTH-1:0] dcr_write_data;
    always @(posedge clock) begin
      if (reset) begin
        dcr_state <= `VX_DCR_ADDR_BITS'h000;
      end else begin
        dcr_state <= dcr_state_n;
      end
    end
    always @(*) begin
      dcr_state_n = dcr_state;
      dcr_write_valid = 1'b0;
      dcr_write_addr = `VX_DCR_ADDR_WIDTH'b0;
      dcr_write_data = `VX_DCR_DATA_WIDTH'b0;

      case (dcr_state)
        `VX_DCR_ADDR_BITS'h000: begin
          dcr_state_n = `VX_DCR_BASE_STATE_BEGIN;
        end
        `VX_DCR_BASE_STATE_BEGIN: begin
          dcr_state_n = `VX_DCR_BASE_STARTUP_ADDR1;

          dcr_write_valid = 1'b1;
          dcr_write_addr = `VX_DCR_BASE_STARTUP_ADDR0;
          dcr_write_data = BOOTROM_HANG100;
        end
        `VX_DCR_BASE_STARTUP_ADDR1: begin
          dcr_state_n = `VX_DCR_BASE_MPM_CLASS;

          dcr_write_valid = 1'b1;
          dcr_write_addr = `VX_DCR_BASE_STARTUP_ADDR1;
          // FIXME: not sure what this does
          dcr_write_data = `VX_DCR_DATA_WIDTH'h0;
        end
        `VX_DCR_BASE_MPM_CLASS: begin
          dcr_state_n = `VX_DCR_BASE_STATE_END;

          dcr_write_valid = 1'b1;
          dcr_write_addr = `VX_DCR_BASE_MPM_CLASS;
          dcr_write_data = `VX_DCR_DATA_WIDTH'h0;
        end
        `VX_DCR_BASE_STATE_END: begin
          dcr_state_n = dcr_state;

          dcr_write_valid = 1'b0;
        end
      endcase
    end

    VX_dcr_bus_if dcr_bus_if();
    assign dcr_bus_if.write_valid = dcr_write_valid;
    assign dcr_bus_if.write_addr  = dcr_write_addr;
    assign dcr_bus_if.write_data  = dcr_write_data;

    VX_core #(
        .CORE_ID (CORE_ID)
    ) core (
        `SCOPE_IO_BIND  (0) // TODO: should be socket id

        .clk            (clock),
        .reset          (core_reset),

    `ifdef PERF_ENABLE
        // NOTE unused
        .mem_perf_if    (mem_perf_tmp_if),
    `endif
        
        .dcr_bus_if     (dcr_bus_if),

        .dcache_bus_if  (dcache_bus_if),

        .icache_bus_if  (icache_bus_if),

    `ifdef GBAR_ENABLE
        // NOTE unused
        .gbar_bus_if    (per_core_gbar_bus_if[i]),
    `endif

        .sim_ebreak     (sim_ebreak),
        .sim_wb_value   (sim_wb_value),
        .busy           (busy)
    );

    // VX_dcache_req_if #(
    //     .NUM_REQS  (`DCACHE_NUM_REQS),
    //     .WORD_SIZE (`DCACHE_WORD_SIZE),
    //     .TAG_WIDTH (`DCACHE_CORE_TAG_WIDTH)
    // ) dcache_req_if();

    // VX_dcache_rsp_if #(
    //     .NUM_REQS  (`DCACHE_NUM_REQS),
    //     .WORD_SIZE (`DCACHE_WORD_SIZE), 
    //     .TAG_WIDTH (`DCACHE_CORE_TAG_WIDTH)
    // ) dcache_rsp_if();
    // 
    // VX_icache_req_if #(
    //     .WORD_SIZE (`ICACHE_WORD_SIZE), 
    //     .TAG_WIDTH (`ICACHE_CORE_TAG_WIDTH)
    // ) icache_req_if();

    // VX_icache_rsp_if #(
    //     .WORD_SIZE (`ICACHE_WORD_SIZE), 
    //     .TAG_WIDTH (`ICACHE_CORE_TAG_WIDTH)
    // ) icache_rsp_if();

    // VX_pipeline #(
    //     .CORE_ID(CORE_ID)
    // ) pipeline (
    //     `SCOPE_BIND_VX_core_pipeline
    // `ifdef PERF_ENABLE
    //     .perf_memsys_if (perf_memsys_if),
    // `endif

    //     .clk(clock),
    //     .reset(reset || intr_reset),

    //     .irq(1'b0/*intr_reset*/),

    //     // Dcache core request
    //     .dcache_req_valid   (dcache_req_if.valid),
    //     .dcache_req_rw      (dcache_req_if.rw),
    //     .dcache_req_byteen  (dcache_req_if.byteen),
    //     .dcache_req_addr    (dcache_req_if.addr),
    //     .dcache_req_data    (dcache_req_if.data),
    //     .dcache_req_tag     (dcache_req_if.tag),
    //     .dcache_req_ready   (dcache_req_if.ready),

    //     // Dcache core reponse    
    //     .dcache_rsp_valid   (dcache_rsp_if.valid),
    //     .dcache_rsp_tmask   (dcache_rsp_if.tmask),
    //     .dcache_rsp_data    (dcache_rsp_if.data),
    //     .dcache_rsp_tag     (dcache_rsp_if.tag),
    //     .dcache_rsp_ready   (dcache_rsp_if.ready),

    //     // Icache core request
    //     .icache_req_valid   (icache_req_if.valid),
    //     .icache_req_addr    (icache_req_if.addr),
    //     .icache_req_tag     (icache_req_if.tag),
    //     .icache_req_ready   (icache_req_if.ready),

    //     // Icache core reponse    
    //     .icache_rsp_valid   (icache_rsp_if.valid),
    //     .icache_rsp_data    (icache_rsp_if.data),
    //     .icache_rsp_tag     (icache_rsp_if.tag),
    //     .icache_rsp_ready   (icache_rsp_if.ready),

    //     // Status
    //     .busy(busy)
    // );

    always @(*) begin
        if (busy === 1'b0) begin
            $display("---------------- no more active warps ----------------");

            @(negedge clock);

            // TODO: lane assumed to be 4
            // `ifndef SYNTHESIS
            // for (integer j = 0; j < `NUM_WARPS; j++) begin
            //     $display("warp %2d", j);
            //     for (integer k = 0; k < `NUM_REGS; k += 1)
            //         $display("x%2d: %08x  %08x  %08x  %08x", k,
            //             pipeline.issue.gpr_stage.iports[/*thread*/0].dp_ram1.not_out_reg.reg_dump.ram[j * `NUM_REGS + k],
            //             pipeline.issue.gpr_stage.iports[/*thread*/1].dp_ram1.not_out_reg.reg_dump.ram[j * `NUM_REGS + k],
            //             pipeline.issue.gpr_stage.iports[/*thread*/2].dp_ram1.not_out_reg.reg_dump.ram[j * `NUM_REGS + k],
            //             pipeline.issue.gpr_stage.iports[/*thread*/3].dp_ram1.not_out_reg.reg_dump.ram[j * `NUM_REGS + k]);
            //     end
            // `endif

            // @(posedge clock) $finish();
        end
    end

endmodule : Vortex





