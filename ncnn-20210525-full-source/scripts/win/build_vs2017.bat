cd ../../
mkdir vs2017
cd vs2017
cmake -G "Visual Studio 15 2017" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" -DProtobuf_INCLUDE_DIR=D:/library/protobuf-3.4.0/vs2017/install/include -DProtobuf_LIBRARIES=D:/library/protobuf-3.4.0/vs2017/install/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=D:/library/protobuf-3.4.0/vs2017/install/bin/protoc.exe ..

pause
