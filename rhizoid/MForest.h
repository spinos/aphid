#pragma once
#include "DrawForest.h"
#include <maya/MGlobal.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MMatrix.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>
#include <maya/MDoubleArray.h>
#include <boost/scoped_array.hpp>

namespace aphid {

class ForestCell;

/// maya interface
class MForest : public DrawForest {
	
	boost::scoped_array<int> m_randGroup;
	
public:
    MForest();
    virtual ~MForest();
    
	void selectPlantByType(const MPoint & origin, const MPoint & dest,  int typ,
					MGlobal::ListAdjustment adj);
	void finishPlantSelection();
	void selectGround(const MPoint & origin, const MPoint & dest, 
					MGlobal::ListAdjustment adj);
	void finishGroundSelection();
/// cover selected faces
	void flood(GrowOption & option);
	void grow(const MPoint & origin, const MPoint & dest, 
					GrowOption & option);
	void replacePlant(const MPoint & origin, const MPoint & dest, 
					GrowOption & option);
	void erase(const MPoint & origin, const MPoint & dest,
					GrowOption & option);
    void adjustBrushSize(const MPoint & origin, const MPoint & dest, 
                    float magnitude);
	void adjustSize(const MPoint & origin, const MPoint & dest, 
                     float magnitude,
					 bool isBundled);
	void adjustRotation(const MPoint & origin, const MPoint & dest, 
                        float magnitude, short axis);
	void extractActive(int numGroups);
	
	void loadExternal(const char* filename);
	void saveExternal(const char* filename);
	void deselectFaces();
    void deselectPlants();
    void injectPlants(const std::vector<Matrix44F> & ms, GrowOption & option);
	void offsetAlongNormal(const MPoint & origin, const MPoint & dest,
					GrowOption & option);
	void movePlantByVec(const Ray & ray,
						const Vector3F & displaceNear, const Vector3F & displaceFar,
						const float & clipNear, const float & clipFar);
    
protected:
    bool updateGround(MArrayDataHandle & meshDataArray, MArrayDataHandle & spaceDataArray);
	static void matrix_as_array(const MMatrix &space, double *mm);
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
						MDoubleArray & replacers,
						const int & numGroups);
	void updateExamples(MArrayDataHandle & dataArray);
	void pickVisiblePlants(float lodLowGate, float lodHighGate, 
					double percentage,
                    int plantTyp);	
				
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
					unsigned & it);
    void saveActiveExternal(const char* filename);
    void saveAllExternel(const char* filename);
    void getDataRef(PlantData * plt, 
					const int & plantTyp,
					float * data, 
					int * typd,
					unsigned & it);
	void pickupVisiblePlantsInCell(ForestCell *cell,
					float lodLowGate, float lodHighGate, 
					double percentage, int plantTyp, 
                    int & it);
					
};

}
