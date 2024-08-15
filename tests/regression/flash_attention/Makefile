PROJECT = flash_attention

SRCS = main.cpp common.h

VX_SRCS = kernel.cpp
VX_INCLUDES = ../sgemm_tcore/sgemm_impl.hpp

OPTS ?= -n16

VX_CFLAGS += -I../sgemm_tcore

include ../common.mk
