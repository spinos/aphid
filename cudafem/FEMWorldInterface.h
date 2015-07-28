#ifndef FEMWORLDINTERFACE_H
#define FEMWORLDINTERFACE_H

#include <DynamicWorldInterface.h>
#include <StripeMap.h>
class FEMTetrahedronSystem;
class FEMWorldInterface : public DynamicWorldInterface {
public:
    FEMWorldInterface();
    virtual ~FEMWorldInterface();
    
    virtual void create(CudaDynamicWorld * world);
    bool useVelocityFile(CudaDynamicWorld * world);
protected:

private:
	bool readTetrahedronMeshFromFile(CudaDynamicWorld * world);
	bool readTriangleMeshFromFile(CudaDynamicWorld * world);
    StripeMap m_map;
};

#endif        //  #ifndef FEMWORLDINTERFACE_H

