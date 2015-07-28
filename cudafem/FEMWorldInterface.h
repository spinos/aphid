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
    
    void updateStiffnessMapEnds(float a, float b);
    void updateStiffnessMapLeft(float x, float y);
    void updateStiffnessMapRight(float x, float y);
	void updateDensity(float x);
	void updateYoungsModulus(float x);
protected:

private:
	bool readTetrahedronMeshFromFile(CudaDynamicWorld * world);
	bool readTriangleMeshFromFile(CudaDynamicWorld * world);
};

#endif        //  #ifndef FEMWORLDINTERFACE_H

