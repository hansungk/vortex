all:
	$(MAKE) -C kernel
	$(MAKE) -C tests

clean:
	$(MAKE) -C kernel clean
	$(MAKE) -C tests clean

clean-all:
	$(MAKE) -C kernel clean
	$(MAKE) -C tests clean-all

