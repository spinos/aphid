#include <iostream>
#include "TutorialConfig.h"
#include <hdf5.h>
int main (int argc, char * const argv[]) {
    // insert code here...
    std::cout << "Hello, CMake!\n";
    std::cout<<"CMake Tutorial Version "<<Tutorial_VERSION_MAJOR<<"."<<Tutorial_VERSION_MINOR;
    std::cout<<"HDF5 test\n";
    hid_t fFileId = H5Fcreate("dest.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    std::cout<<"file id is "<<fFileId;
    return 0;
}
