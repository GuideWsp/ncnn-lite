* [Build for Linux x86](#build-for-linux-x86)
* [Build for Raspberry Pi 3](#build-for-raspberry-pi-3)
* [Build for ARM Cortex-A family with cross-compiling](#build-for-arm-cortex-a-family-with-cross-compiling)
* [Build for Hisilicon platform with cross-compiling](#build-for-hisilicon-platform-with-cross-compiling)

***

### Build for Linux x86
install g++ cmake protobuf

```
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j4
```

***

### Build for Raspberry Pi 3
install g++ cmake protobuf
```
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake -DPI3=ON ..
$ make -j4
$ make install
```

pick build/install folder for further usage

***

### Build for ARM Cortex-A family with cross-compiling
download ARM toolchain from https://developer.arm.com/open-source/gnu-toolchain/gnu-a/downloads
```
$ export PATH=<your-toolchain-compiler-path>:$PATH
```
AArch32 target with soft float (arm-linux-gnueabi)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-arm-linux-gnueabi
$ cd build-arm-linux-gnueabi
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabi.toolchain.cmake ..
$ make -j4
$ make install
```
AArch32 target with hard float (arm-linux-gnueabihf)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-arm-linux-gnueabihf
$ cd build-arm-linux-gnueabihf
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-linux-gnueabihf.toolchain.cmake ..
$ make -j4
$ make install
```
AArch64 GNU/Linux target (aarch64-linux-gnu)
```
$ cd <ncnn-root-dir>
$ mkdir -p build-aarch64-linux-gnu
$ cd build-aarch64-linux-gnu
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
$ make -j4
$ make install
```

pick build-XXXXX/install folder for further usage

***

### Build for Hisilicon platform with cross-compiling
download and install Hisilicon SDK
```
# the path that toolchain should be installed in
$ ls /opt/hisi-linux/x86-arm
```
```
$ cd <ncnn-root-dir>
$ mkdir -p build
$ cd build

# choose one cmake toolchain file depends on your target platform
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv300.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/hisiv500.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix100.toolchain.cmake ..
$ cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/himix200.toolchain.cmake ..

$ make -j4
$ make install
```

pick build/install folder for further usage
