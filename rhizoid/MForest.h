#pragma once
#include "DrawForest.h"
#include "GroupSelect.h"
#include <maya/MGlobal.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MMatrix.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>
#include <maya/MDoubleArray.h>

namespace aphid {

class ForestCell;

/// maya interface
class MForest : public DrawForest, public GroupSelect {
	
public:
    MForest();
    virtual ~MForest();
    
	void selectPlantByType(const MPoint & origin, const MPoint & dest,  int typ,
					MGlobal::ListAdjustment adj);
	void finishPlantSelection();
	void selectGround(const MPoint & origin, const MPoint & dest, 
					MGlobal::ListAdjustment adj);
	void finishGroundSelection();

	void grow(const MPoint & origin, const MPoint & dest, 
					GrowOption & option);
	void replacePlant(const MPoint & origin, const MPoint & dest, 
					GrowOption & option);
	void erase(const MPoint & origin, const MPoint & dest,
					GrowOption & option);
	void beginAdjustBrushSize(const MPoint & origin, const MPoint & dest,
					GrowOption & option);
    void adjustBrushSize(float magnitude);
	void adjustSize(const MPoint & origin, const MPoint & dest, 
                     float magnitude,
					 bool isBundled);
	void adjustRotation(const MPoint & origin, const MPoint & dest, 
                        float magnitude, short axis);
	void extractActive(int numGroups);
	
	bool loadExternal(const char* filename);
	bool saveExternal(const char* filename);
	void deselectPlants();
    void injectPlants(const std::vector<Matrix44F> & ms, GrowOption & option);
	void offsetAlongNormal(GrowOption & option);
	void movePlantByVec(const Ray & ray,
						const Vector3F & displaceNear, const Vector3F & displaceFar,
						const float & clipNear, const float & clipFar);
    
protected:
    bool updateGround(MArrayDataHandle & meshDataArray, MArrayDataHandle & spaceDataArray);
	void savePlants(MPointArray & plantTms, 
					MIntArray & plantIds,
					MIntArray & plantTris,
					MVectorArray & plantCoords,
					MVectorArray & plantOffsets);
	bool loadPlants(const MPointArray & plantTms, 
					const MIntArray & plantIds,
					const MIntArray & plantTris,
					const MVectorArray & plantCoords,
					MVectorArray & plantOffsets);
	void bakePass(const char* filename, 
					const MVectorArray & position, 
					const MVectorArray & scale, 
					const MVectorArray & rotation);
	void initRandGroup();
/// Per-Particle attribs for instancer
	void computePPAttribs(MVectorArray & positions,
						MVectorArray & rotations,
						MVectorArray & scales,
						MDoubleArray & replacers);
	void updateExamples(MArrayDataHandle & dataArray);
	void pickVisiblePlants(float lodLowGate, float lodHighGate, 
					double percentage,
                    int plantTyp);	
	void flood(GrowOption & option);
	
private:
    void updateGroundMesh(MObject & mesh, const MMatrix & worldTm, unsigned idx);
    void saveCell(ForestCell *cell,
					MPointArray & plantTms, 
					MIntArray & plantIds,
					MIntArray & plantTris,
					MVectorArray & plantCoords,
					MVectorArray & plantOffsets);
	void getDataInCell(ForestCell *cell, 
					float * data, 
					int * typd,
					float * voff,
					unsigned & it);
    bool saveActiveExternal(const char* filename);
    bool saveAllExternel(const char* filename);
    void getDataRef(PlantData * plt, 
					const int & plantTyp,
					float * data, 
					int * typd,
					float * voff,
					unsigned & it);
	void pickupVisiblePlantsInCell(ForestCell *cell,
					float lodLowGate, float lodHighGate, 
					double percentage, int plantTyp, 
                    int & it);
	void addExampleAttribs(const int & iExample,
					PlantInstance * pli,
					MVectorArray & positions,
					MVectorArray & rotations,
					MVectorArray & scales,
					MDoubleArray & replacers);
/// multi instances for compount 
	void addCompoundExampleAttribs(const ExampVox * exmp,
					PlantInstance * pli,
					MVectorArray & positions,
					MVectorArray & rotations,
					MVectorArray & scales,
					MDoubleArray & replacers);
	void appendAInstance(MVectorArray & positions,
					MVectorArray & rotations,
					MVectorArray & scales,
					MDoubleArray & replacers,
					const Matrix44F & mat,
					const int & instanceId);
					
};

}
