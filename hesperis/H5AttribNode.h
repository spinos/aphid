#include <maya/MPlug.h>
#include <maya/MPxNode.h>
#include <maya/MDataBlock.h>
#include <maya/MObject.h> 
#include <maya/MFnNumericData.h>
#include <maya/MString.h>
#include <EnvVar.h>
#include <H5Holder.h>

class H5AttribNode : public MPxNode, public EnvVar, public H5Holder
{
public:
						H5AttribNode();
	virtual				~H5AttribNode(); 

	virtual MStatus		compute( const MPlug& plug, MDataBlock& data );

	static  void*		creator();
	static  MStatus		initialize();

    virtual MStatus connectionMade(const MPlug &plug, const MPlug &otherPlug, bool asSrc);
	
public:
	static  	MObject		input;
	static  MObject aframe;
	static MObject aminframe;
	static MObject amaxframe;
	static MObject abyteAttrName;
	static  	MObject 	outByte;
	static MObject ashortAttrName;
	static  	MObject 	outShort;
	static MObject aintAttrName;
	static  	MObject 	outInt;
	static MObject afloatAttrName;
	static  	MObject 	outFloat;
	static MObject adoubleAttrName;
	static  	MObject 	outDouble;
	static MObject aboolAttrName;
	static  	MObject 	outBool;
	
	static	MTypeId		id;
	
private:
	static void createNameValueAttr(MObject & nameAttr, MObject & valueAttr,
						const MString & name1L, const MString & name1S, 
						const MString & name2L, const MString & name2S, 
						MFnNumericData::Type valueTyp);
};


