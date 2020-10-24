cd ../../
mkdir vs2017
cd vs2017
cmake -G "Visual Studio 15 2017" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" ..

pause