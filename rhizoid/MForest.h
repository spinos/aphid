#pragma once
#include "DrawForest.h"
#include <maya/MGlobal.h>
#include <maya/MArrayDataHandle.h>

/// maya interface
class MForest : public DrawForest {

public:
    MForest();
    virtual ~MForest();
    
	void selectGround(const MPoint & origin, const MPoint & dest, 
					MGlobal::ListAdjustment adj);
	void flood(GrowOption & option);
	
protected:
    void updateGround(MArrayDataHandle & data);
	
private:
    void updateGroundMesh(MObject & mesh, unsigned idx);
    
};