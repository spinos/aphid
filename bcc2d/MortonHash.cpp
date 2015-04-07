#include "MortonHash.h"

namespace sdb {
CellHash::CellHash() {}
CellHash::~CellHash() {}

EdgeHash::EdgeHash() {}
EdgeHash::~EdgeHash() {}

void swap(uint & a, uint & b)
{
    if(a > b) {
        uint c = a;
        a = b;
        b = c;
    }
}

uint64 upsample(uint a, uint b) 
{ return ((uint64)a << 32) | (uint64)b; }

void downsample(uint64 combined, uint & a, uint & b)
{
    a = combined >> 32;
    b = combined & ~0x80000000;
}

void EdgeHash::addEdge(unsigned a, unsigned b)
{
    swap(a, b);
    uint64 c = upsample(a, b);
    EdgeValue * v = new EdgeValue;
    v->level = 10;
    insert(c, v);
}

void EdgeHash::connectedTo(unsigned & a, unsigned & b)
{
    downsample(key(), a, b);
}

} // end namespace sdb
