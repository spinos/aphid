#ifndef MORTONHASH_H
#define MORTONHASH_H

#include <Array.h>

namespace sdb {

struct CellValue {
	int level;
	float a;
};

class MortonHash : public Array<unsigned, CellValue>
{
public:
    MortonHash() {}
    virtual ~MortonHash() {}
    
protected:

private:

};
} // end namespace sdb
#endif        //  #ifndef MORTONHASH_H

