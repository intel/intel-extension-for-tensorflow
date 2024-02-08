target = label_image
backend = <BACKEND>
cc = g++
TF_INCLUDE_PATH = <TF_PATH>/include/
TFCC_PATH = <TF_PATH>
ITEX_CC_PATH= <ITEX_PATH>
include = -I $(TF_INCLUDE_PATH)

ifeq ($(backend), GPU)
    lib = -L $(TFCC_PATH) -L $(ITEX_CC_PATH) -lintel_xla -ltensorflow_framework -ltensorflow_cc
else
    lib = -L $(TFCC_PATH) -ltensorflow_framework -ltensorflow_cc
endif

flag = -Wl,-rpath=$(TFCC_PATH) -std=c++17 -D<BUILD_TARGET>
source = ./label_image.cc
$(target): $(source)
	$(cc) $(source) -o $(target) $(include) $(lib) $(flag)
clean:
	rm $(target)
run:
	./$(target)
