/*
 *  SolverNode.h
 *  caterpillar
 *
 *  Created by jian zhang on 3/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>
#include <maya/MTime.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <AllMath.h>
#include "PhysicsState.h"

namespace caterpillar {
class SolverNode : public MPxLocatorNode, public PhysicsState
{
public:
	SolverNode();
	virtual ~SolverNode(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 

	static  void *          creator();
	static  MStatus         initialize();

	static  MObject a_inTime;
	static  MObject a_startTime;
	static  MObject a_gravity;
	static  MObject a_enable;
	static  MObject a_numSubsteps;
	static  MObject a_frequency;
	static MObject a_timeScale;
	static  MObject a_inConditions;
	static  MObject a_outRigidBodies;
	static	MTypeId id;
	
private:
	void computeConditions(MDataBlock& block);
	MTime m_preTime;
};
}
