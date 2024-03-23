// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

`ifdef EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

module VX_core import VX_gpu_pkg::*; #( 
    parameter CORE_ID = 0
) (        
    `SCOPE_IO_DECL
    
    // Clock
    input wire              clk,
    input wire              reset,

`ifdef PERF_ENABLE
    VX_mem_perf_if.slave    mem_perf_if,
`endif

    VX_dcr_bus_if.slave     dcr_bus_if,

    VX_mem_bus_if.master    smem_bus_if [DCACHE_NUM_REQS],

    VX_mem_bus_if.master    dcache_bus_if [DCACHE_NUM_REQS],

    VX_mem_bus_if.master    icache_bus_if,

`ifdef GBAR_ENABLE
    VX_gbar_bus_if.master   gbar_bus_if,
`endif

    // simulation helper signals
    output wire             sim_ebreak,
    output wire [`NUM_REGS-1:0][`XLEN-1:0] sim_wb_value,

    // Status
    output wire             busy    //stays 1 when busy, 0 when done (termination) detect the negative edge
);
    VX_schedule_if      schedule_if();
    VX_fetch_if         fetch_if();
    VX_decode_if        decode_if();
    VX_sched_csr_if     sched_csr_if();
    VX_decode_sched_if  decode_sched_if();
    VX_commit_sched_if  commit_sched_if();
    VX_commit_csr_if    commit_csr_if();
    VX_branch_ctl_if    branch_ctl_if[`NUM_ALU_BLOCKS]();
    VX_warp_ctl_if      warp_ctl_if();    
    
    VX_dispatch_if      alu_dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if        alu_commit_if[`ISSUE_WIDTH]();

    VX_dispatch_if      lsu_dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if        lsu_commit_if[`ISSUE_WIDTH]();
`ifdef EXT_F_ENABLE 
    VX_dispatch_if      fpu_dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if        fpu_commit_if[`ISSUE_WIDTH]();
`endif
    VX_dispatch_if      sfu_dispatch_if[`ISSUE_WIDTH]();
    VX_commit_if        sfu_commit_if[`ISSUE_WIDTH]();    
    
    VX_writeback_if     writeback_if[`ISSUE_WIDTH]();

    VX_mem_bus_if #(
        .DATA_SIZE (DCACHE_WORD_SIZE), 
        .TAG_WIDTH (DCACHE_TAG_WIDTH)
    ) dcache_bus_tmp_if[DCACHE_NUM_REQS]();

`ifdef PERF_ENABLE
    VX_mem_perf_if mem_perf_tmp_if();
    VX_pipeline_perf_if pipeline_perf_if();    

    assign mem_perf_tmp_if.icache  = mem_perf_if.icache;
    assign mem_perf_tmp_if.dcache  = mem_perf_if.dcache;
    assign mem_perf_tmp_if.l2cache = mem_perf_if.l2cache;
    assign mem_perf_tmp_if.l3cache = mem_perf_if.l3cache;
    assign mem_perf_tmp_if.mem     = mem_perf_if.mem;
`endif

    `RESET_RELAY (dcr_data_reset, reset);
    `RESET_RELAY (schedule_reset, reset);
    `RESET_RELAY (fetch_reset, reset);
    `RESET_RELAY (decode_reset, reset);
    `RESET_RELAY (issue_reset, reset);
    `RESET_RELAY (execute_reset, reset);
    `RESET_RELAY (commit_reset, reset);

    base_dcrs_t base_dcrs;

    VX_dcr_data dcr_data (
        .clk        (clk),
        .reset      (dcr_data_reset),
        .dcr_bus_if (dcr_bus_if),
        .base_dcrs  (base_dcrs)
    );

    `SCOPE_IO_SWITCH (3)

    VX_schedule #(
        .CORE_ID (CORE_ID)
    ) schedule (
        .clk            (clk),
        .reset          (schedule_reset),

    `ifdef PERF_ENABLE
        .perf_schedule_if (pipeline_perf_if.schedule),
    `endif 

        .base_dcrs      (base_dcrs),  

        .warp_ctl_if    (warp_ctl_if),        
        .branch_ctl_if  (branch_ctl_if),
        .decode_sched_if(decode_sched_if),
        .commit_sched_if(commit_sched_if),

        .schedule_if    (schedule_if),
    `ifdef GBAR_ENABLE
        .gbar_bus_if    (gbar_bus_if),
    `endif
        .sched_csr_if   (sched_csr_if),        

        .busy           (busy)
    );

    VX_fetch #(
        .CORE_ID (CORE_ID)
    ) fetch (
        `SCOPE_IO_BIND  (0)
        .clk            (clk),
        .reset          (fetch_reset),
        .icache_bus_if  (icache_bus_if),
        .schedule_if    (schedule_if),
        .fetch_if       (fetch_if)
    );

    VX_decode #(
        .CORE_ID (CORE_ID)
    ) decode (
        .clk            (clk),
        .reset          (decode_reset),
        .fetch_if       (fetch_if),
        .decode_if      (decode_if),
        .decode_sched_if(decode_sched_if)
    );

    VX_issue #(
        .CORE_ID (CORE_ID)
    ) issue (
        `SCOPE_IO_BIND  (1)

        .clk            (clk),
        .reset          (issue_reset),

    `ifdef PERF_ENABLE
        .perf_issue_if  (pipeline_perf_if.issue),
    `endif

        .decode_if      (decode_if),
        .writeback_if   (writeback_if),

        .alu_dispatch_if(alu_dispatch_if),
        .lsu_dispatch_if(lsu_dispatch_if),
    `ifdef EXT_F_ENABLE
        .fpu_dispatch_if(fpu_dispatch_if),
    `endif
        .sfu_dispatch_if(sfu_dispatch_if)
    );

    VX_execute #(
        .CORE_ID (CORE_ID)
    ) execute (
        `SCOPE_IO_BIND  (2)
        
        .clk            (clk),
        .reset          (execute_reset),

        .base_dcrs      (base_dcrs),

    `ifdef PERF_ENABLE
        .mem_perf_if    (mem_perf_tmp_if),        
        .pipeline_perf_if(pipeline_perf_if),
    `endif 

        .dcache_bus_if  (dcache_bus_tmp_if),
    
    `ifdef EXT_F_ENABLE
        .fpu_dispatch_if(fpu_dispatch_if),
        .fpu_commit_if  (fpu_commit_if),
    `endif

        .commit_csr_if  (commit_csr_if),
        .sched_csr_if   (sched_csr_if),
        
        .alu_dispatch_if(alu_dispatch_if),
        .lsu_dispatch_if(lsu_dispatch_if),
        .sfu_dispatch_if(sfu_dispatch_if),

        .warp_ctl_if    (warp_ctl_if),
        .branch_ctl_if  (branch_ctl_if),

        .alu_commit_if  (alu_commit_if),
        .lsu_commit_if  (lsu_commit_if),
        .sfu_commit_if  (sfu_commit_if),

        .sim_ebreak     (sim_ebreak)
    );    

    VX_commit #(
        .CORE_ID (CORE_ID)
    ) commit (
        .clk            (clk),
        .reset          (commit_reset),

        .alu_commit_if  (alu_commit_if),
        .lsu_commit_if  (lsu_commit_if),
    `ifdef EXT_F_ENABLE
        .fpu_commit_if  (fpu_commit_if),
    `endif
        .sfu_commit_if  (sfu_commit_if),
        
        .writeback_if   (writeback_if),
        
        .commit_csr_if  (commit_csr_if),
        .commit_sched_if(commit_sched_if),

        .sim_wb_value   (sim_wb_value)
    );

`ifdef SM_ENABLE

    VX_smem_unit #(
        .CORE_ID (CORE_ID)
    ) smem_unit (
        .clk                (clk),
        .reset              (reset),
    `ifdef PERF_ENABLE
        .cache_perf         (mem_perf_tmp_if.smem),
    `endif
        .dcache_bus_in_if   (dcache_bus_tmp_if),
        .dcache_bus_out_if  (dcache_bus_if),
        .smem_bus_out_if    (smem_bus_if)
    );

`else

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        `ASSIGN_VX_MEM_BUS_IF (dcache_bus_if[i], dcache_bus_tmp_if[i]);
    end

`endif

`ifdef PERF_ENABLE  // expose these perf counter to console using $display, %time; flag: --perf=0?

    wire [`CLOG2(DCACHE_NUM_REQS+1)-1:0] perf_dcache_rd_req_per_cycle;
    wire [`CLOG2(DCACHE_NUM_REQS+1)-1:0] perf_dcache_wr_req_per_cycle;
    wire [`CLOG2(DCACHE_NUM_REQS+1)-1:0] perf_dcache_rsp_per_cycle;    

    wire [1:0] perf_icache_pending_read_cycle;
    wire [`CLOG2(DCACHE_NUM_REQS+1)+1-1:0] perf_dcache_pending_read_cycle;

    reg [`PERF_CTR_BITS-1:0] perf_icache_pending_reads;
    reg [`PERF_CTR_BITS-1:0] perf_dcache_pending_reads;

    reg [`PERF_CTR_BITS-1:0] perf_ifetches;
    reg [`PERF_CTR_BITS-1:0] perf_loads;
    reg [`PERF_CTR_BITS-1:0] perf_stores;

    wire perf_icache_req_fire = icache_bus_if.req_valid && icache_bus_if.req_ready;
    wire perf_icache_rsp_fire = icache_bus_if.rsp_valid && icache_bus_if.rsp_ready;

    wire [DCACHE_NUM_REQS-1:0] perf_dcache_rd_req_fire, perf_dcache_rd_req_fire_r;
    wire [DCACHE_NUM_REQS-1:0] perf_dcache_wr_req_fire, perf_dcache_wr_req_fire_r;
    wire [DCACHE_NUM_REQS-1:0] perf_dcache_rsp_fire;

    for (genvar i = 0; i < DCACHE_NUM_REQS; ++i) begin
        assign perf_dcache_rd_req_fire[i] = dcache_bus_if[i].req_valid && dcache_bus_if[i].req_ready && ~dcache_bus_if[i].req_data.rw;
        assign perf_dcache_wr_req_fire[i] = dcache_bus_if[i].req_valid && dcache_bus_if[i].req_ready && dcache_bus_if[i].req_data.rw;
        assign perf_dcache_rsp_fire[i] = dcache_bus_if[i].rsp_valid && dcache_bus_if[i].rsp_ready;
    end

    `BUFFER(perf_dcache_rd_req_fire_r, perf_dcache_rd_req_fire);
    `BUFFER(perf_dcache_wr_req_fire_r, perf_dcache_wr_req_fire);

    `POP_COUNT(perf_dcache_rd_req_per_cycle, perf_dcache_rd_req_fire_r);
    `POP_COUNT(perf_dcache_wr_req_per_cycle, perf_dcache_wr_req_fire_r);
    `POP_COUNT(perf_dcache_rsp_per_cycle, perf_dcache_rsp_fire);
      
    assign perf_icache_pending_read_cycle = perf_icache_req_fire - perf_icache_rsp_fire;
    assign perf_dcache_pending_read_cycle = perf_dcache_rd_req_per_cycle - perf_dcache_rsp_per_cycle;

    always @(posedge clk) begin
        if (reset) begin
            perf_icache_pending_reads <= '0;
            perf_dcache_pending_reads <= '0;
        end else begin
            perf_icache_pending_reads <= $signed(perf_icache_pending_reads) + `PERF_CTR_BITS'($signed(perf_icache_pending_read_cycle));
            perf_dcache_pending_reads <= $signed(perf_dcache_pending_reads) + `PERF_CTR_BITS'($signed(perf_dcache_pending_read_cycle));
        end
    end
    
    reg [`PERF_CTR_BITS-1:0] perf_icache_lat;
    reg [`PERF_CTR_BITS-1:0] perf_dcache_lat;

    always @(posedge clk) begin
        if (reset) begin
            perf_ifetches   <= '0;
            perf_loads      <= '0;
            perf_stores     <= '0;
            perf_icache_lat <= '0;
            perf_dcache_lat <= '0;
        end else begin
            perf_ifetches   <= perf_ifetches   + `PERF_CTR_BITS'(perf_icache_req_fire);
            perf_loads      <= perf_loads      + `PERF_CTR_BITS'(perf_dcache_rd_req_per_cycle);
            perf_stores     <= perf_stores     + `PERF_CTR_BITS'(perf_dcache_wr_req_per_cycle);
            perf_icache_lat <= perf_icache_lat + perf_icache_pending_reads;
            perf_dcache_lat <= perf_dcache_lat + perf_dcache_pending_reads;
        end
    end

    assign pipeline_perf_if.ifetches = perf_ifetches;
    assign pipeline_perf_if.loads = perf_loads;
    assign pipeline_perf_if.stores = perf_stores;
    assign pipeline_perf_if.load_latency = perf_dcache_lat;
    assign pipeline_perf_if.ifetch_latency = perf_icache_lat;
    int instrs;
    assign instrs = 32'(commit_csr_if.instret);
    int cycles;
    assign cycles = 32'(sched_csr_if.cycles);
    int icache_lat;
    assign icache_lat = 32'(perf_icache_lat);
    int ifetches;
    assign ifetches = 32'(perf_ifetches);
    int dcache_lat;
    assign dcache_lat = 32'(perf_dcache_lat);
    int loads;
    assign loads = 32'(perf_loads);
    int scrb_alu_per_core;
    assign scrb_alu_per_core = 32'(pipeline_perf_if.units_uses[`EX_ALU]);
    int scrb_fpu_per_core;
    assign scrb_fpu_per_core = 32'(pipeline_perf_if.units_uses[`EX_FPU]);
    int scrb_lsu_per_core;
    assign scrb_lsu_per_core = 32'(pipeline_perf_if.units_uses[`EX_LSU]);
    int scrb_sfu_per_core;
    assign scrb_sfu_per_core = 32'(pipeline_perf_if.units_uses[`EX_SFU]);
    int scrb_tot;
    assign scrb_tot = scrb_alu_per_core+scrb_fpu_per_core+scrb_lsu_per_core+scrb_sfu_per_core;
    int scrb_wctl_per_core;
    assign scrb_wctl_per_core = 32'(pipeline_perf_if.sfu_uses[`SFU_WCTL]);
    int scrb_csrs_per_core;
    assign scrb_csrs_per_core = 32'(pipeline_perf_if.sfu_uses[`SFU_CSRS]);
    int sfu_tot;
    assign sfu_tot = scrb_wctl_per_core+scrb_csrs_per_core;
    
    reg busy_prev;
    reg [31:0] report_counter;

    always @(posedge clk) begin
      if (reset) begin
        busy_prev <= 1'b0;
        report_counter <= 32'd0;
      end else begin
        busy_prev <= busy;
        if (report_counter == 32'd10000) begin
            report_counter <= 32'd0;
        end else begin
            report_counter <= report_counter + 32'd1;
        end
      end
    end

    wire busy_negedge;
    assign busy_negedge = busy_prev && !busy;

    reg [`PERF_CTR_BITS-1:0] dispatch_fires_total;
    always @(*) begin
        dispatch_fires_total = '0;
        for (integer i = 0; i < `NUM_EX_UNITS; i++) begin
            dispatch_fires_total = dispatch_fires_total + pipeline_perf_if.dispatch_fires[i];
        end
    end

    always @(posedge clk) begin
        if (!reset && (busy_negedge || (report_counter == 32'd0))) begin
            $display("====================CORE : %d===================",CORE_ID);
            $display("time : %t", $time);
            // disabled as always zero
            // $display("perf_dcache_rd_req_per_cycle: %d", perf_dcache_rd_req_per_cycle);
            // $display("perf_dcache_wr_req_per_cycle: %d", perf_dcache_wr_req_per_cycle);
            // $display("perf_dcache_rsp_per_cycle: %d", perf_dcache_rsp_per_cycle);
            // $display("perf_icache_pending_read_cycle: %d", perf_icache_pending_read_cycle);
            // $display("perf_dcache_pending_read_cycle: %d", perf_dcache_pending_read_cycle);
            // $display("perf_icache_pending_reads: %d", perf_icache_pending_reads);
            // $display("perf_dcache_pending_reads: %d", perf_dcache_pending_reads);
            // $display("perf_icache_req_fire: %b", perf_icache_req_fire);
            // $display("perf_icache_rsp_fire: %b", perf_icache_rsp_fire);
            // $display("perf_dcache_rd_req_fire: %b", perf_dcache_rd_req_fire);
            // $display("perf_dcache_rd_req_fire_r: %b", perf_dcache_rd_req_fire_r);
            // $display("perf_dcache_wr_req_fire: %b", perf_dcache_wr_req_fire);
            // $display("perf_dcache_wr_req_fire_r: %b", perf_dcache_wr_req_fire_r);
            // $display("perf_dcache_rsp_fire: %b", perf_dcache_rsp_fire);

            $display("Instructions: %d, Cycles: %d, IPC: %f", commit_csr_if.instret, sched_csr_if.cycles,
                     $itor(instrs) / $itor(cycles));
            $display("scheduler idle: %d cycles (%.2f%%)", pipeline_perf_if.sched_idles,
                     $itor(pipeline_perf_if.sched_idles) / $itor(cycles) * 100.0);
            $display("scheduler stalls: %d cycles (%.2f%%)", pipeline_perf_if.sched_stalls,
                     $itor(pipeline_perf_if.sched_stalls) / $itor(cycles) * 100.0);
            $display("scheduler barrier stalls: %d count across NUM_WARPS=%d (%.2f%%)",
                     pipeline_perf_if.sched_barrier_stalls,
                     `NUM_WARPS,
                     $itor(pipeline_perf_if.sched_barrier_stalls) / $itor(cycles) * 100.0);
            $display("decode stalls: %d cycles (%.2f%%)",pipeline_perf_if.ibf_stalls,
                     $itor(pipeline_perf_if.ibf_stalls) / $itor(cycles) * 100.0);
            // see VX_scoreboard.sv
            // scb_stalls: valid & ~ready (ready = stg_ready_in && operands_ready)
            // units_uses: valid & ~operands_ready
            //             this will be a subset of scb_stalls
            $display("issue scoreboard: fires total:\t%d across ISSUE_WIDTH=%d",
                     pipeline_perf_if.scb_fires, `ISSUE_WIDTH);
            $display("issue scoreboard: stalls total:\t%d across ISSUE_WIDTH=%d",
                     pipeline_perf_if.scb_stalls, `ISSUE_WIDTH);
            $display("issue scoreboard: stalls by operand hazard: alu %d (%2.2f cycles per issue)",
                     scrb_alu_per_core,
                     $itor(scrb_alu_per_core) / $itor(dispatch_fires_total));
            $display("issue scoreboard: stalls by operand hazard: fpu %d (%2.2f cycles per issue)",
                     scrb_fpu_per_core,
                     $itor(scrb_fpu_per_core) / $itor(dispatch_fires_total));
            $display("issue scoreboard: stalls by operand hazard: lsu %d (%2.2f cycles per issue)",
                     scrb_lsu_per_core,
                     $itor(scrb_lsu_per_core) / $itor(dispatch_fires_total));
            $display("issue scoreboard: stalls by operand hazard: sfu %d (%2.2f cycles per issue)",
                     scrb_sfu_per_core,
                     $itor(scrb_sfu_per_core) / $itor(dispatch_fires_total));
            $display("issue scoreboard: sfu stalls: %d (scrs=%f, wctl=%f)",pipeline_perf_if.units_uses[`EX_SFU],
                     $itor(scrb_csrs_per_core) / $itor(sfu_tot) * 100.0,
                     $itor(scrb_wctl_per_core) / $itor(sfu_tot) * 100.0);
            $display("issue dispatch: stalls by FU busy: alu %d (%2.2f cycles per issue)",
                     pipeline_perf_if.dispatch_stalls[`EX_ALU],
                     $itor(pipeline_perf_if.dispatch_stalls[`EX_ALU]) / $itor(dispatch_fires_total));
            $display("issue dispatch: stalls by FU busy: fpu %d (%2.2f cycles per issue)",
                     pipeline_perf_if.dispatch_stalls[`EX_FPU],
                     $itor(pipeline_perf_if.dispatch_stalls[`EX_FPU]) / $itor(dispatch_fires_total));
            $display("issue dispatch: stalls by FU busy: lsu %d (%2.2f cycles per issue)",
                     pipeline_perf_if.dispatch_stalls[`EX_LSU],
                     $itor(pipeline_perf_if.dispatch_stalls[`EX_LSU]) / $itor(dispatch_fires_total));
            $display("issue dispatch: stalls by FU busy: sfu %d (%2.2f cycles per issue)",
                     pipeline_perf_if.dispatch_stalls[`EX_SFU],
                     $itor(pipeline_perf_if.dispatch_stalls[`EX_SFU]) / $itor(dispatch_fires_total));
            $display("issue dispatch: fires: total %d",
                     dispatch_fires_total);
            $display("issue dispatch: fires: alu %d",
                     pipeline_perf_if.dispatch_fires[`EX_ALU]);
            $display("issue dispatch: fires: fpu %d",
                     pipeline_perf_if.dispatch_fires[`EX_FPU]);
            $display("issue dispatch: fires: lsu %d",
                     pipeline_perf_if.dispatch_fires[`EX_LSU]);
            $display("issue dispatch: fires: sfu %d",
                     pipeline_perf_if.dispatch_fires[`EX_SFU]);
            $display("issue dispatch: cycles issued: %d (%.2f%%)",
                     pipeline_perf_if.dispatch_any_fire_cycles,
                     $itor(pipeline_perf_if.dispatch_any_fire_cycles) / $itor(cycles) * 100.0);
            $display("ifetches: %d", perf_ifetches);
            $display("ifetch latency: %f Cycles",
                     $itor(icache_lat) / $itor(ifetches));
            $display("loads: %d", perf_loads);
            $display("load latency: %f Cycles",
                     $itor(dcache_lat) / $itor(loads));
            $display("stores: %d", perf_stores);
        end
    end

`endif

endmodule
