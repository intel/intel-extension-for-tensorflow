target = label_image
cc = g++
TF_INCLUDE_PATH = <TF_PATH>/include/
TFCC_PATH = <TF_PATH>
include = -I $(TF_INCLUDE_PATH)
lib = -L $(TFCC_PATH) -ltensorflow_framework -ltensorflow_cc
flag = -Wl,-rpath=$(TFCC_PATH) -std=c++17 -D<BUILD_TARGET>
source = ./label_image.cc
$(target): $(source)
	$(cc) $(source) -o $(target) $(include) $(lib) $(flag)
clean:
	rm $(target)
run:
	./$(target)