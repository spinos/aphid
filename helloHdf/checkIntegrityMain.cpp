#include <HFile.h>
#include <iostream>
using namespace aphid;

int main (int argc, char * const argv[]) {
    if(argc < 2) {
        std::cout << " Check H5 integrity has no input";
		return 1;
	}
	
    std::cout << "\n Check H5 integrity of "<<argv[1];
	
    if(!HObject::FileIO.open(argv[1], HDocument::oReadOnly)) {
		std::cout << "\n cannot open file, check failed. return 1";
        return 1;
	}
    
    HObject::FileIO.close();
    std::cout << "\n passed. return 0";
	return 0;
}
