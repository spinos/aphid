#pragma once
#include "Forest.h"
#include <maya/MArrayDataHandle.h>

class MForest : public sdb::Forest {

public:
    MForest();
    virtual ~MForest();
    
protected:
    void updateGround(MArrayDataHandle & data);
	
private:
    void updateGroundMesh(MObject & mesh, unsigned idx);
    
};
