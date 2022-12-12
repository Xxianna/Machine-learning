PKG_sse4 :=  -fopenmp -msse4
PKG :=  -fopenmp
tensorpath := libqwqtensor
MY := $(tensorpath)\libqwqtensor.lib
MY_sse4 := $(tensorpath)\libqwqtensor_SSE4.lib



qwqnn: libqwqnn.cpp libqwqnn.hpp layers.cpp op_lossfunc.cpp
	g++ -c op_lossfunc.cpp -o op_lossfunc.o $(PKG)
	g++ -c layers.cpp -o layers.o $(PKG)
	g++ -c libqwqnn.cpp -o libqwqnn.o $(PKG) 
	ar rcs libqwqnn.lib libqwqnn.o layers.o op_lossfunc.o
	del *.o

qwqnnsse: libqwqnn.cpp libqwqnn.hpp layers.cpp op_lossfunc.cpp
	g++ -c op_lossfunc.cpp -o op_lossfunc.o $(PKG_sse4)
	g++ -c layers.cpp -o layers.o $(PKG_sse4)
	g++ -c libqwqnn.cpp -o libqwqnn.o $(PKG_sse4) 
	ar rcs libqwqnn_SSE4.lib libqwqnn.o layers.o op_lossfunc.o
	del *.o

# test : tensortest.cpp libqwqtensor.hpp accelerator.h
# 	g++ -o tensortest.exe tensortest.cpp $(PKG) $(MY)
# 	.\tensortest.exe

# testsse : tensortest.cpp libqwqtensor.hpp accelerator.h
# 	g++ -o tensortestsse.exe tensortest.cpp $(PKG_sse4) $(MY_sse4)
# 	.\tensortestsse.exe