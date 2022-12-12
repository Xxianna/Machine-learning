PKG_sse4 :=  -fopenmp -msse4
PKG :=  -fopenmp
nnpath := libqwqnn
tensorpath := libqwqtensor
MY := $(nnpath)\libqwqnn.lib $(nnpath)\$(tensorpath)\libqwqtensor.lib
MY_sse4 := $(nnpath)\libqwqnn_SSE4.lib $(nnpath)\$(tensorpath)\libqwqtensor_SSE4.lib

hw: main.cpp
	g++ -o hw.exe main.cpp $(PKG) $(MY)
	.\hw.exe

hwsse: main.cpp
	g++ -o hwsse.exe main.cpp $(PKG_sse4) $(MY_sse4)
	.\hwsse.exe
