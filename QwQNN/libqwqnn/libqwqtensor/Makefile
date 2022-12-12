PKG_sse4 :=  -fopenmp -msse4
PKG :=  -fopenmp

MY := libqwqtensor.lib
MY_sse4 := libqwqtensor_SSE4.lib

tensor: libqwqtensor.cpp libqwqtensor.hpp accelerator.h
	g++ -c libqwqtensor.cpp -o libqwqtensor.o $(PKG)
	ar rcs libqwqtensor.lib libqwqtensor.o
	del *.o

tensorsse: libqwqtensor.cpp libqwqtensor.hpp accelerator.h
	g++ -c libqwqtensor.cpp -o libqwqtensor.o $(PKG_sse4)
	ar rcs libqwqtensor_SSE4.lib libqwqtensor.o
	del *.o

# test : tensortest.cpp libqwqtensor.hpp accelerator.h
# 	g++ -o tensortest.exe tensortest.cpp $(PKG) $(MY)
# 	.\tensortest.exe

# testsse : tensortest.cpp libqwqtensor.hpp accelerator.h
# 	g++ -o tensortestsse.exe tensortest.cpp $(PKG_sse4) $(MY_sse4)
# 	.\tensortestsse.exe