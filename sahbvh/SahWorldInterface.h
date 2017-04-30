/*
 *  SahWorldInterface.h
 *  testsah
 *
 *  Created by jian zhang on 5/3/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#include <DynamicWorldInterface.h>
class CudaDynamicWorld;
class SahTetrahedronSystem;
class SahWorldInterface : public DynamicWorldInterface {
public:
	SahWorldInterface();
	virtual ~SahWorldInterface();
	
	virtual void create(CudaDynamicWorld * world);
protected:

private:
	bool readMeshFromFile(SahTetrahedronSystem * mesh);
	void createTestMesh(SahTetrahedronSystem * mesh);
	void resetVelocity(SahTetrahedronSystem * mesh);
};
