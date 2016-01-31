#pragma once
#include "DrawForest.h"
#include <maya/MGlobal.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MMatrix.h>

/// maya interface
class MForest : public DrawForest {

public:
    MForest();
    virtual ~MForest();
    
	void selectPlant(const MPoint & origin, const MPoint & dest, 
					MGlobal::ListAdjustment adj);
	void selectGround(const MPoint & origin, const MPoint & dest, 
					MGlobal::ListAdjustment adj);
	void flood(GrowOption & option);
	void grow(const MPoint & origin, const MPoint & dest, 
					GrowOption & option);
	void finishGrow();
	void erase(const MPoint & origin, const MPoint & dest,
					float weight);
	void finishErase();
	
protected:
    void updateGround(MArrayDataHandle & data);
	void drawSolidMesh(MItMeshPolygon & iter);
	void drawWireMesh(MItMeshPolygon & iter);
	static void matrix_as_array(const MMatrix &space, double *mm);
private:
    void updateGroundMesh(MObject & mesh, unsigned idx);
    
};
