#ifndef _HesUVNode
#define _HesUVNode

#include <maya/MPxNode.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <maya/MFloatArray.h>
#include <maya/MIntArray.h>
#include <vector>
#include <EnvVar.h>
 
class HesUVNode : public MPxNode, public EnvVar
{
public:
						HesUVNode();
	virtual				~HesUVNode(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static  void*		creator();
	static  MStatus		initialize();

public:

	static	MTypeId		id;
	static  	MObject		ainput;
	static MObject ameshname;
	static  MObject		ainMesh;
	static  MObject		aoutMesh;
	
private:
		
};

#endif
