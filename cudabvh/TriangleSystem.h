#ifndef TRIANGLESYSTEM_H
#define TRIANGLESYSTEM_H

#include <CudaMassSystem.h>
#include <ATriangleMesh.h>

class TriangleSystem : public CudaMassSystem {
public:
    TriangleSystem(ATriangleMesh * md);
    virtual ~TriangleSystem();

// override mass system
	virtual const int elementRank() const;
    virtual const unsigned numElements() const;
protected:
    
private:
    
};
#endif        //  #ifndef TRIANGLESYSTEM_H

