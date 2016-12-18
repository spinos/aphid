#ifndef _A_HELPER_H
#define _A_HELPER_H

#include <maya/MFnDependencyNode.h>
#include <maya/MPlug.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnVectorArrayData.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MObject.h>
#include <maya/MString.h>
#include <maya/MStatus.h>
#include <maya/MGlobal.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MSelectionList.h>
#include <maya/MItSelectionList.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MMatrix.h>
#include <maya/MIntArray.h>
#include <maya/MPointArray.h>
#include <maya/MVectorArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MTime.h>
#include <maya/MDGContext.h>
#include <maya/MItDag.h>
#include <maya/MItDependencyNodes.h>
#include <maya/MItDependencyGraph.h>
#include <maya/MFnCamera.h>
#include <maya/MFnPluginData.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnMesh.h>
#include <maya/MItMeshPolygon.h>
#include <maya/MPlugArray.h>
#include <maya/MFnTransform.h>
#include <maya/MArgList.h>
#include <sstream>
#undef min
#undef max
#include <boost/lexical_cast.hpp>
#include <math/Matrix44F.h>

namespace aphid {

class AHelper
{
public:
	AHelper(void);
	~AHelper(void);

	static void getColorAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& r, double& g, double& b);
	static void getNormalAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& r, double& g, double& b);
	static char getDoubleAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& v);
	static char getBoolAttributeByName(const MFnDependencyNode& fnode, const char* attrname, bool& v);
	static char getDoubleAttributeByNameAndTime(const MFnDependencyNode& fnode, const char* attrname, MDGContext & ctx, double& v);
	static char getStringAttributeByName(const MFnDependencyNode& fnode, const char* attrname, MString& v);
	static char getStringAttributeByName(const MObject& node, const char* attrname, MString& v);
	static int getConnectedAttributeByName(const MFnDependencyNode& fnode, const char* attrname, MString& v);
	
	static void getNamedPlug(MPlug& val, const MObject& node, const char* attrname);
	static void getIntFromNamedPlug(int& val, const MObject& node, const char* attrname);
	static void getDoubleFromNamedPlug(double& val, const MObject& node, const char* attrname);
	static void getVectorArrayFromPlug(MVectorArray& array, MPlug& plug);
	static void getDoubleArrayFromPlug(MDoubleArray& array, MPlug& plug);
	static void extractMeshParams(const MObject& mesh, unsigned & numVertex, unsigned & numPolygons, MPointArray& vertices, MIntArray& pcounts, MIntArray& pconnects);
	
	static MStatus createIntAttr(MObject& attr, const char* nameLong, const char* nameShort, int val, int vmin);
	static MStatus createDoubleAttr(MObject& attr, const MString& nameLong, const MString& nameShort, double val);
	static MStatus createTypedAttr(MObject& attr, const MString& nameLong, const MString& nameShort, MFnData::Type type);
	static MStatus createTypedArrayAttr(MObject& attr, const MString& nameLong, const MString& nameShort, MFnData::Type type);
	static MStatus createMatrixAttr(MObject& attr, const MString& nameLong, const MString& nameShort);
	static MStatus createStringAttr(MObject& attr, const MString& nameLong, const MString& nameShort);
	static MStatus createTimeAttr(MObject& attr, const MString& nameLong,const MString& nameShort, double val);
	static MStatus createTimeAttrInternal(MObject& attr, const MString& nameLong,const MString& nameShort, double val);
	static MStatus createVectorAttr(MObject& attr, MObject& attr0, MObject& attr1, MObject& attr2, const MString& nameLong,const MString& nameShort);
	static MStatus createVectorArrayAttr(MObject& attr, const MString& nameLong,const MString& nameShort);
	
	static MObject getMeshAttr(MDataBlock& data, MObject& attr);
	
	static void getTypedPath(MFn::Type type, const MObject& root, MDagPath& path);
	static void getAllTypedPath(MFn::Type type, MObjectArray& obj_array);
	static int getAllTypedPathByRoot(MFn::Type type, MObject& root, MObjectArray& obj_array);
	static void getTypedPath(MFn::Type type, const MDagPath& root, MDagPath& path);
	static void getTypedPathByName(MFn::Type type, MString& name, MDagPath& path);
	static void getTypedNodeByName(MFn::Type type, MString& name, MObject& node);
	static MObject getTypedObjByFullName(MFn::Type type, const char* name);
	static char findObjChildByName(MObject &parent, MObject &result, std::string &name);
	static char getObjByFullName(const char* name, MObject& res);
	static char getConnectedPlug(MPlug& val, const MPlug& plg);
	static void getConnectedNode(MObject& val, const MPlug& plg);
	static void getConnectedNodeName(MString& val, const MPlug& plg);
	
	static void getNamedObject(MString& name, MObject& obj);
	
	static MMatrix getMatrixAttr(const MObject& node, MObject& attr);
	static MString getStringAttr(const MObject& node, MObject& attr);
	
	static MVectorArray getVectorArrayAttr(MDataBlock& data, MObject& attr);
	static MDoubleArray getDoubleArrayAttr(MDataBlock& data, MObject& attr);
	static void validateFilePath(MString& name);
	static void noDotDagPath(MString& name);
	static void getProjectDataPath(MString& path);
	static void getWindowsPath(MString& path);
	static void getFileNameFirstDot(MString& name);
	static void cutFileNameLastSlash(MString& name);
	static void changeFileNameExtension(MString& name, const char* ext);
	static int hasNamedAttribute(const MObject& node, const char* attrname);
	static void displayIntParam(const char* str, int val);
	static void displayVectorParam(const char* str, double x, double y, double z);
	
	static void getMat(const MMatrix &mat, float space[16]);
	static void getTransformMat(const MDagPath &path, float space[16]);
	static void getTransformWorld(const MString& name, float space[4][4]);
	static MVector getTransformWorldNoScale(const MString& name, float space[4][4]);
	
	static void getTypedDepNode(MFn::Type type, MObject& root, MObject& node);
	static void getTypedDepNodeByName(MFn::Type type, MString& name, MObject& root, MObject& node);

	static MTime::Unit getPlaybackFPS();
	static MObject createMinimalMesh(MObject& parent, const char* name);
	
	static char findNamedPlugsInHistory(MObject &node, MFn::Type type, MString &name1, MPlug &plug1, MString &name2, MPlug &plug2);
	static void convertToMatrix(MMatrix& mat, float mm[4][4]);
	static void convertToMatrix(float *m, float mm[4][4]);
	
	static char findNamedDagInList(MDagPathArray &arr, const char *name);
	static void createMinimalMaterial(MString &name, MString &texture);
	
	static char containsGeom(const MDagPath & root);
    static std::string FullPathNameToObj(const MObject & obj);
	
	static MMatrix GetWorldParentTransformMatrix(const MDagPath & path);
	static MMatrix GetParentTransform(const MDagPath & path);
/// relative to instanced transform
	static MMatrix GetWorldParentTransformMatrix2(const MDagPath & path,
                    const MDagPath & transformPath);
/// include itself
	static MMatrix GetWorldTransformMatrix(const MDagPath & path);
	static double AreaOfTriangle(const MPoint & a, const MPoint & b, const MPoint & c);
	
	template<typename T>
	static void Info(const std::string & note, const T & v)
	{
		std::stringstream sst;
		sst<<note<<" "<<v;
		MGlobal::displayInfo(sst.str().c_str());
	}
    
    static void PrintMatrix(const std::string & note, const MMatrix & mat);
    static void ConvertToMMatrix(MMatrix & dst, const Matrix44F & src);
	static void ConvertToMatrix44F(Matrix44F & dst, const MMatrix & src);
	static void SimpleAnimation(const MPlug & dst, int a, int b);
	
	template<typename T>
	static void Append(T & dst, const T & src)
	{
		unsigned i, j;
		const unsigned n = src.length();
		for(i=0; i<n; i++) {
			dst.append(src[i]);
		}
	}
	
	template<typename T>
	static void Merge(T & dst, const T & src)
	{
		unsigned i, j;
		const unsigned n = src.length();
		for(i=0; i<n; i++) {
			bool found = false;
			for(j=0; j< dst.length(); j++) {
				if(dst[j] == src[i]) {
					found = true;
					break;
				}
			}
			
			if(!found) dst.append(src[i]);
		}
	}
	
	template<typename T>
	static T GetValueFromStr(const char * src)
	{
		try {
			return boost::lexical_cast<T>(src);
		}
		catch(boost::bad_lexical_cast &)
		{
			return 0;
		}
	}
	
	template<typename T1, typename T2>
	static bool BelongsTo(const T1 & a, const T2 & b) 
	{
	    const unsigned n = b.length();
		for(unsigned i=0; i<n; i++) {
			
			if(a == b[i]) {
			    return true;
			}
		}
		return false;
	}
    
    static bool IsReferenced(const MObject & node);
	static MObject CreateDeformer(const MString & name);
	static void PrintFullPath(const MDagPathArray & paths);
	static MDagPath FindMatchedPath(const MDagPath & rel,
	                const MDagPathArray & paths);
	static void GetInputConnections(MPlugArray & dst, const MPlug & p);
	static void GetOutputConnections(MPlugArray & dst, const MPlug & p);
	static void GetArrayPlugInputConnections(MPlugArray & dst, const MPlug & p);
	static bool GetDepNodeByName(MObject & dst, MFn::Type type, const MString & name);
	static bool GetStringArgument(MString & res,
	                            unsigned & it,
	                            const unsigned & argc,
	                            const MArgList &args,
	                            const std::string & log1,
	                            const std::string & log2);
/// first in array not connected
	static void GetAvailablePlug(MPlug & dst, MPlug & p);
	
};

}
#endif
