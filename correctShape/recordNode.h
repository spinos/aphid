#include <maya/MPxNode.h> 

class PoseRecordNode : public MPxNode
{
public:
						PoseRecordNode();
	virtual				~PoseRecordNode(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static  void*		creator();
	static  MStatus		initialize();

public:
	static  MObject		output;        // The output value.
	static MObject aposespace;
	static	MTypeId		id;
};