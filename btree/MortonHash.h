#ifndef MORTONHASH_H
#define MORTONHASH_H

#include <Array.h>

namespace aphid {

namespace sdb {
    
typedef unsigned uint;
typedef unsigned long long uint64;

struct CellValue {
	int level;
	int visited;
	unsigned index;
};

class CellHash : public Array<unsigned, CellValue>
{
public:
    CellHash();
    virtual ~CellHash();
    
protected:

private:

};

struct EdgeValue {
    int visited;
    float a;
};

class EdgeHash : public Array<uint64, EdgeValue>
{
public:
    EdgeHash();
    virtual ~EdgeHash();
    EdgeValue * findEdge(unsigned a, unsigned b);
    void addEdge(unsigned a, unsigned b);
    void connectedTo(unsigned & a, unsigned & b);
protected:

private:

};

} // end namespace sdb

}
#endif        //  #ifndef MORTONHASH_H

