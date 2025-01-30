all: regression # opencl

regression:
	$(MAKE) -C regression

opencl:
ifneq ($(XLEN),64)
	$(MAKE) -C opencl
endif

clean:
	$(MAKE) -C regression clean
	$(MAKE) -C opencl clean

clean-all:
	$(MAKE) -C regression clean-all
	# $(MAKE) -C opencl clean-all

.PHONY: all regression opencl
