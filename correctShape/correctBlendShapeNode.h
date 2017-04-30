#include <maya/MGlobal.h>
#include <maya/MPxNode.h> 
#include <maya/MPointArray.h>
#include <maya/MFloatArray.h>
#include <maya/MMatrixArray.h>

class CorrectBlendShapeNode : public MPxNode
{
public:
						CorrectBlendShapeNode();
	virtual				~CorrectBlendShapeNode(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static  void*		creator();
	static  MStatus		initialize();

public:
	static  	MObject 	asculptMesh;
	static  	MObject 	aposefile;
	static MObject aspacerow0;
	static MObject aspacerow1;
	static MObject aspacerow2;
	static MObject aspacerow3;
	static MObject abindpnt;
	static MObject aposepnt;
	static  	MObject 	outMesh;
	static	MTypeId		id;
	
private:
	char loadCache(const char* filename, unsigned numVertex);
	char readInternalCache(MDataBlock& block, unsigned numVertex);
	MPointArray _bindPoseVertex;
	MPointArray _sculptPoseVertex;
	MMatrixArray _poseSpace;
	char _isCached;
};


