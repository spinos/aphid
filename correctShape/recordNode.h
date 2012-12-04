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
	static MObject aposespacerow0;
	static MObject aposespacerow1;
	static MObject aposespacerow2;
	static MObject aposespacerow3;
	static MObject abindpnt;
	static MObject aposepnt;
	static	MTypeId		id;
};