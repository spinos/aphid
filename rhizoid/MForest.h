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
	void adjustSize(const MPoint & origin, const MPoint & dest, 
                     float magnitude);
	void adjustRotation(const MPoint & origin, const MPoint & dest, 
                        float magnitude, short axis);
	void extractActive(int numGroups);

protected:
    void updateGround(MArrayDataHandle & data);
	void drawSolidMesh(MItMeshPolygon & iter);
	void drawWireMesh(MItMeshPolygon & iter);
	static void matrix_as_array(const MMatrix &space, double *mm);
	void savePlants(MPointArray & plantTms, 
					MIntArray & plantIds,
					MIntArray & plantTris,
					MVectorArray & plantCoords);
	bool loadPlants(const MPointArray & plantTms, 
					const MIntArray & plantIds,
					const MIntArray & plantTris,
					const MVectorArray & plantCoords);
	void loadExternal(const char* filename);
	void saveExternal(const char* filename);
	void bakePass(const char* filename, 
					const MVectorArray & position, 
					const MVectorArray & scale, 
					const MVectorArray & rotation);
	
private:
    void updateGroundMesh(MObject & mesh, unsigned idx);
    void saveCell(sdb::Array<int, sdb::Plant> *cell,
					MPointArray & plantTms, 
					MIntArray & plantIds,
					MIntArray & plantTris,
					MVectorArray & plantCoords);
};
