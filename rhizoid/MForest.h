#pragma once
#include "DrawForest.h"
#include <maya/MGlobal.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MMatrix.h>

/// maya interface
class MForest : public DrawForest {
	
	int * m_randGroup;
	
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
	void replacePlant(const MPoint & origin, const MPoint & dest, 
					GrowOption & option);
	void finishGrow();
	void erase(const MPoint & origin, const MPoint & dest,
					GrowOption & option);
	void finishErase();
	void adjustSize(const MPoint & origin, const MPoint & dest, 
                     float magnitude);
	void adjustRotation(const MPoint & origin, const MPoint & dest, 
                        float magnitude, short axis);
	void extractActive(int numGroups);
	
	void loadExternal(const char* filename);
	void saveExternal(const char* filename);
	
protected:
    bool updateGround(MArrayDataHandle & meshDataArray, MArrayDataHandle & spaceDataArray);
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
	void bakePass(const char* filename, 
					const MVectorArray & position, 
					const MVectorArray & scale, 
					const MVectorArray & rotation);
	void initRandGroup();
	void pickVisiblePlants(bool hasCamera, float lodLowGate, float lodHighGate, 
					int totalGroups, int currentGroup, 
					double percentage);
	void saveParticles(MVectorArray & positions,
						MVectorArray & rotations,
						MVectorArray & scales);
	void updateExamples(MArrayDataHandle & dataArray);	
				
private:
    void updateGroundMesh(MObject & mesh, const MMatrix & worldTm, unsigned idx);
    void saveCell(sdb::Array<int, sdb::Plant> *cell,
					MPointArray & plantTms, 
					MIntArray & plantIds,
					MIntArray & plantTris,
					MVectorArray & plantCoords);
	void getDataInCell(sdb::Array<int, sdb::Plant> *cell, 
					float * data, unsigned & it);
	void pickupVisiblePlantsInCell(sdb::Array<int, sdb::Plant> *cell,
					bool hasCamera, float lodLowGate, float lodHighGate, 
					int totalGroups, int currentGroup, 
					double percentage, int & it);
};
