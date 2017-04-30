#include <maya/MPlug.h>
#include <maya/MPxNode.h>
#include <maya/MDataBlock.h>
#include <maya/MObject.h> 
#include <EnvVar.h>

class HesMeshNode : public MPxNode, public EnvVar
{
public:
						HesMeshNode();
	virtual				~HesMeshNode(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static  void*		creator();
	static  MStatus		initialize();

    virtual MStatus connectionMade(const MPlug &plug, const MPlug &otherPlug, bool asSrc);
	
public:
	static  	MObject		input;
	static MObject ameshname;
	static  	MObject 	outMesh;
	static	MTypeId		id;
	
private:
};


