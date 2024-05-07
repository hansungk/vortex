num_sets = 4
num_steps = 4
num_substeps = 2


def set_step_substep(sequence_number):
    set_num, step = divmod(sequence_number, num_steps * num_substeps)
    step //= num_substeps
    substep = sequence_number % 2

    return set_num, step, substep

# set + substep -> rs1, rs2
rs1 = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 2,
    (1, 1): 3,
    (2, 0): 4,
    (2, 1): 5,
    (3, 0): 6,
    (3, 1): 7 
}

rs2 = {
    (0, 0): 8,
    (0, 1): 9,
    (1, 0): 10,
    (1, 1): 11,
    (2, 0): 12,
    (2, 1): 13,
    (3, 0): 14,
    (3, 1): 15 
}

# step + substep -> rs3, rd
rs3_rd = {
    (0, 0): 16,
    (0, 1): 17,
    (1, 0): 18,
    (1, 1): 19,
    (2, 0): 20,
    (2, 1): 21,
    (3, 0): 22,
    (3, 1): 23
}

with open('VX_tensor_ucode.vh', 'w') as f:
    for sequence_number in range(num_sets * num_steps * num_substeps):
        set_num, step, substep = set_step_substep(sequence_number)

        
        next_sequence_num = (sequence_number + 1) % (num_sets * num_steps * num_substeps)
        next_set_num, next_step, next_substep = set_step_substep(next_sequence_num)
        finish = (next_sequence_num == 0)

        name = "HMMA_SET{}_STEP{}_{}"
        ucode = "{}, HMMA_SET{}_STEP{}_{}, `EX_BITS'(`EX_TENSOR), `INST_OP_BITS'({}), `INST_MOD_BITS'({}), 1'b1, 1'b0, 1'b0, 32'b{}, 32'b{}, `FREG({}), `FREG({}), `FREG({}), `FREG({})"
        
        name = name.format(
            set_num, step, substep,
        )
        
        pc_imm = 1 if finish else 0

        ucode = ucode.format(
            "FINISH" if finish else "NEXT",
            next_set_num, next_step, next_substep,
            step,
            substep,
            pc_imm,
            pc_imm,
            rs3_rd[(step, substep)],
            rs1[(set_num, substep)],
            rs2[(set_num, substep)],
            rs3_rd[(step, substep)],
        )

        entry = f"{name}: begin \n"
        entry += "\tuop = {"
        entry += ucode
        entry += "}; \n"
        entry += "end \n"

        f.write(entry)