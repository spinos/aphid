#ifndef FEMWORLDINTERFACE_H
#define FEMWORLDINTERFACE_H

#include <DynamicWorldInterface.h>
class FEMTetrahedronSystem;
class FEMWorldInterface : public DynamicWorldInterface {
public:
    FEMWorldInterface();
    virtual ~FEMWorldInterface();
    
    virtual void create(CudaDynamicWorld * world);
    
protected:

private:
	bool readMeshFromFile(FEMTetrahedronSystem * mesh);
	void createTestMesh(FEMTetrahedronSystem * mesh);
	void resetVelocity(FEMTetrahedronSystem * mesh);
};

#endif        //  #ifndef FEMWORLDINTERFACE_H

