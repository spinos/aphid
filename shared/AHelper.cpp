#include "AHelper.h"
#include <AllMath.h>
#include <maya/MFnAnimCurve.h>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>
class MString;
#include <string>
#include <vector>
using namespace std;
#include <sstream>

AHelper::AHelper(void)
{
}

AHelper::~AHelper(void)
{
}

MStatus AHelper::createVectorAttr(MObject& attr, 
					MObject& attr0, MObject& attr1, MObject& attr2, 
					const MString& nameLong,const MString& nameShort)
{
	MStatus status;
	MFnNumericAttribute attrFn;

	attr= attrFn.create(nameLong, nameShort, attr0, attr1, attr2, &status);
	attrFn.setStorable(false);
	attrFn.setKeyable(false);
	attrFn.setReadable(true);
	attrFn.setWritable(false);
	return status;
}

void AHelper::getVectorArrayFromPlug(MVectorArray& array, MPlug& plug)
{
	MObject obj;
	plug.getValue(obj);
	MFnVectorArrayData data(obj);
	array = data.array();
}

void AHelper::getDoubleArrayFromPlug(MDoubleArray& array, MPlug& plug)
{
	MObject obj;
	plug.getValue(obj);
	MFnDoubleArrayData data(obj);
	array = data.array();
}

void AHelper::extractMeshParams(const MObject& mesh, unsigned & numVertex, unsigned & numPolygons, MPointArray& vertices, MIntArray& pcounts, MIntArray& pconnects)
{
	MFnMesh finmesh(mesh);
	numVertex = finmesh.numVertices();
	numPolygons = finmesh.numPolygons();
	finmesh.getPoints(vertices);
	
	pcounts.clear();
	pconnects.clear();
	MItMeshPolygon faceIter(mesh);
	faceIter.reset();
	for( ; !faceIter.isDone(); faceIter.next() ) 
	{
		pcounts.append(faceIter.polygonVertexCount());
		MIntArray  vexlist;
		faceIter.getVertices ( vexlist );
		for( unsigned int i=0; i < vexlist.length(); i++ ) 
		{
			pconnects.append(vexlist[vexlist.length()-1-i]);
		}
	}
}

MStatus AHelper::createIntAttr(MObject& attr, const char* nameLong, const char* nameShort, int val, int min)
{
	MStatus status;
	MFnNumericAttribute fAttr;
	attr = fAttr.create(MString(nameLong), MString(nameShort), MFnNumericData::kInt, val, &status);
	fAttr.setStorable(true);
	fAttr.setReadable(true);
	fAttr.setKeyable(true);
	fAttr.setConnectable(true);
	fAttr.setMin(min);
	return status;
}

MStatus AHelper::createMatrixAttr(MObject& attr, const MString& nameLong, const MString& nameShort)
{
	MStatus status;
	MFnMatrixAttribute fAttr;
	attr = fAttr.create(nameLong, nameShort, MFnMatrixAttribute::kDouble, &status);
	fAttr.setStorable(false);
	fAttr.setReadable(false);
	fAttr.setConnectable(true);
	return status;
}

MStatus AHelper::createStringAttr(MObject& attr, const MString& nameLong, const MString& nameShort)
{
	MStatus status;
	MFnTypedAttribute fAttr;
	attr = fAttr.create(nameLong, nameShort, MFnData::kString, MObject::kNullObj, &status);
	fAttr.setStorable(true);
	fAttr.setReadable(true);
	fAttr.setConnectable(false);
	return status;
}

MStatus AHelper::createVectorArrayAttr(MObject& attr, const MString& nameLong,const MString& nameShort)
{
	MStatus status;
	MFnTypedAttribute fAttr;
	attr = fAttr.create(nameLong, nameShort, MFnData::kVectorArray, MObject::kNullObj, &status);
	fAttr.setStorable(false);
	fAttr.setReadable(true);
	fAttr.setConnectable(true);
	return status;
}

void AHelper::getNamedPlug(MPlug& val, const MObject& node, const char* attrname)
{
	MFnDependencyNode fnode(node );
	val = fnode.findPlug(attrname);
}

void AHelper::getIntFromNamedPlug(int& val, const MObject& node, const char* attrname)
{
	MFnDependencyNode fnode(node );
	MPlug plg = fnode.findPlug(attrname);
	plg.getValue(val);
}

void AHelper::getDoubleFromNamedPlug(double& val, const MObject& node, const char* attrname)
{
	MFnDependencyNode fnode(node );
	MPlug plg = fnode.findPlug(attrname);
	plg.getValue(val);
}

MObject AHelper::getMeshAttr(MDataBlock& data, MObject& attr)
{
	MStatus status;
	MDataHandle hdata = data.inputValue(attr, &status);
	 if ( MS::kSuccess != status ) MGlobal::displayWarning("ERROR getting mesh data handle");
    	return hdata.asMesh();
}

void AHelper::getTypedPath(MFn::Type type, const MObject& root, MDagPath& path)
{	
	MItDag itdag;
	itdag.reset(root, MItDag::kDepthFirst, type);
        
        for(; !itdag.isDone(); itdag.next())
        {
        	if(itdag.item().hasFn(type))
        	{
        		itdag.getPath(path);
        		return;
        	}
        }
}

void AHelper::getTypedPath(MFn::Type type, const MDagPath& root, MDagPath& path)
{
	MItDag itdag;
	itdag.reset(root, MItDag::kDepthFirst, type);
	        
        for(; !itdag.isDone(); itdag.next())
        {
        	if(itdag.item().hasFn(type))
        	{
        		itdag.getPath(path);
        		return;
        	}
        }
}

void AHelper::getAllTypedPath(MFn::Type type, MObjectArray& obj_array)
{
	obj_array.clear();
	
	MStatus stat;
	MItDependencyNodes itdag(type, &stat);
	
	if(!stat) MGlobal::displayInfo("Error creating iterator");
	
	for(; !itdag.isDone(); itdag.next())
        {
		MObject aobj =itdag.thisNode();
		obj_array.append(aobj);
        }
}

int AHelper::getAllTypedPathByRoot(MFn::Type type, MObject& root, MObjectArray& obj_array)
{
	obj_array.clear();
	
	MStatus stat;
	MItDependencyGraph itdag(root, type, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat);
	
	if(!stat) return 0; //MGlobal::displayInfo("Error creating iterator");
	
	for(; !itdag.isDone(); itdag.next())
        {
		MObject aobj =itdag.thisNode();
		obj_array.append(aobj);
        }
	
	return 1;
}

void AHelper::getTypedPathByName(MFn::Type type, MString& name, MDagPath& path)
{	
	MItDag itdag(MItDag::kDepthFirst, type);
	        
        for(; !itdag.isDone(); itdag.next())
        {
        	if(itdag.item().hasFn(type))
        	{
        		itdag.getPath(path);
				
        		MFnDagNode pf(path);
				if(pf.name()==name) return;
        	}
        }
}

void AHelper::getTypedDepNode(MFn::Type type, MObject& root, MObject& node)
{
	node = MObject::kNullObj;
	MStatus stat;
	MItDependencyGraph itdep(root, type, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	        
	for(; !itdep.isDone(); itdep.next())
	{
		node = itdep.currentItem();
		return;
	}
}

void AHelper::getTypedDepNodeByName(MFn::Type type, MString& name, MObject& root, MObject& node)
{	
//MFnDependencyNode rf(root);MGlobal::displayInfo(rf.name());
	node = MObject::kNullObj;
	MStatus stat;
	MItDependencyGraph itdep(root, type, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	        
	//if(!stat) MGlobal::displayInfo("no it");
	//else MGlobal::displayInfo("it it why po1");
        for(; !itdep.isDone(); itdep.next())
        {
        		MFnDependencyNode pf(itdep.currentItem());
				//MGlobal::displayInfo(pf.name());
				if(pf.name()==name) {
					node = itdep.currentItem();
					return;
				}
        	
        }
}

void AHelper::getTypedNodeByName(MFn::Type type, MString& name, MObject& node)
{	
	node = MObject::kNullObj;
	MItDag itdag(MItDag::kDepthFirst, type);
	        
        for(; !itdag.isDone(); itdag.next())
        {
        	if(itdag.item().hasFn(type))
        	{
        		MFnDagNode pf(itdag.currentItem());
				//MGlobal::displayInfo(pf.partialPathName());
				if(pf.partialPathName()==name) 
				{
					node = itdag.currentItem();
					return;
				}
        	}
        }
}

char AHelper::findObjChildByName(MObject &parent, MObject &result, std::string &name)
{
	/*std::stringstream sst;
	sst.str("");
	sst<<"(.*)\\_"<<name;
	const boost::regex expre(sst.str());
	
	std::string name_wo_prefix = name;
	int has_prefix = name_wo_prefix.find('_', 0);
	if(has_prefix > 0)
	{
		name_wo_prefix.erase(0, has_prefix+1);
	}
		
	sst.str("");
	sst<<"(.*)\\_"<<name_wo_prefix;
	const boost::regex expre_wo(sst.str());

	boost::match_results<std::string::const_iterator> what;*/
	if(parent != MObject::kNullObj)
	{
		MFnDagNode ppf(parent);
		for(unsigned i = 0; i <ppf.childCount(); i++)
		{
			MFnDagNode pf(ppf.child(i));
			if(pf.name() == name.c_str()) 
			{
				result = ppf.child(i);
				return 1;
			}
		}
	}
	else
	{
	MItDag itdag(MItDag::kBreadthFirst);
	
	if(parent != MObject::kNullObj)
		itdag.reset(parent);
	
	for(; !itdag.isDone(); itdag.next())
	{
		MFnDagNode pf(itdag.currentItem());
		if(pf.name() == name.c_str()) 
		{
			result = itdag.currentItem();
			return 1;
		}
		/*else
		{
			std::string tomatch(pf.name().asChar());
			
			regex_match(tomatch, what, expre, boost::match_partial );
			
			if(what[0].matched)
			{
				result = itdag.currentItem();
				return 1;
			}
			
			if(has_prefix > 0)
			{
				regex_match(tomatch, what, expre_wo, boost::match_partial );
				
				if(what[0].matched)
				{
					result = itdag.currentItem();
					return 1;
				}
			}
		}*/
	}
	}
	result = MObject::kNullObj;
	return 0;
}

char AHelper::getObjByFullName(const char* name, MObject& res)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	MObject parent = MObject::kNullObj;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		if(!AHelper::findObjChildByName( parent, res, r))
			return 0;
		
		parent = res;
	}
	return 1;
}

MObject AHelper::getTypedObjByFullName(MFn::Type type, const char* name)
{
	MItDag itdag(MItDag::kDepthFirst, type);
	        
	for(; !itdag.isDone(); itdag.next())
	{
		MFnDagNode pf(itdag.currentItem());
		if(pf.fullPathName() == MString(name)) 
		{
			return itdag.currentItem();
		}
	}
	return MObject::kNullObj;
}

char AHelper::getConnectedPlug(MPlug& val, const MPlug& plg)
{
	MPlugArray conns;
	if(!plg.connectedTo (conns, true, true )) return 0;
	else val = conns[0];
	return 1;
}

void AHelper::getConnectedNode(MObject& val, const MPlug& plg)
{
	MPlugArray conns;
	if(!plg.connectedTo (conns, true, true )) val = MObject::kNullObj;
	else val = conns[0].node();
}

void AHelper::getConnectedNodeName(MString& val, const MPlug& plg)
{
	MObject obj;
	MPlugArray conns;
	if(!plg.connectedTo (conns, true, true ))  obj = MObject::kNullObj;
	else  obj = conns[0].node();
	
	if(obj != MObject::kNullObj) val = MFnDagNode(obj).fullPathName();
}

MStatus AHelper::createTypedArrayAttr(MObject& attr, const MString& nameLong, const MString& nameShort, MFnData::Type type)
{
	MStatus status;
	MFnTypedAttribute fAttr;
	attr = fAttr.create(nameLong, nameShort, type, &status);
	fAttr.setStorable(false);
	fAttr.setReadable(true);
	fAttr.setConnectable(true);
	fAttr.setArray(true);
	return status;
}

void AHelper::getNamedObject(MString& name, MObject& obj) 
{
	MGlobal::selectByName(name, MGlobal::kReplaceList);
	
	MSelectionList activeList;
	MGlobal::getActiveSelectionList(activeList);
	if(activeList.length()<1)
	{
		obj = MObject::kNullObj;
	}
	
	MItSelectionList iter(activeList);
	iter.getDependNode(obj);
	
//MGlobal::unselectByName(name);
}

MVectorArray AHelper::getVectorArrayAttr(MDataBlock& data, MObject& attr)
{
	MStatus status;
	MDataHandle hdata = data.inputValue(attr, &status);
	
	//if ( MS::kSuccess != status ) MGlobal::displayWarning("ERROR getting vector array data handle");
    	
	MFnVectorArrayData farray(hdata.data(), &status);
	
	//if ( MS::kSuccess != status ) MGlobal::displayWarning("ERROR creating vector array data array");
    
    	return farray.array();
}

MDoubleArray AHelper::getDoubleArrayAttr(MDataBlock& data, MObject& attr)
{
	MStatus status;
	MDataHandle hdata = data.inputValue(attr, &status);
	
	//if ( MS::kSuccess != status ) MGlobal::displayWarning("ERROR getting double array data handle");
    
	MFnDoubleArrayData farray(hdata.data(), &status);
	
	//if ( MS::kSuccess != status ) MGlobal::displayWarning("ERROR creating double array data array");
    
    	return farray.array();
}

MStatus AHelper::createTypedAttr(MObject& attr, const MString& nameLong, const MString& nameShort, MFnData::Type type)
{
	MStatus status;
	MFnTypedAttribute fAttr;
	attr = fAttr.create(nameLong, nameShort, type, &status);
	fAttr.setStorable(false);
	fAttr.setReadable(true);
	fAttr.setConnectable(true);
	return status;
}

MStatus AHelper::createTimeAttr(MObject& attr, const MString& nameLong,const MString& nameShort, double val)
{
	MStatus status;
	MFnUnitAttribute unitAttr;
	attr = unitAttr.create(nameLong, nameShort, MFnUnitAttribute::kTime, val, &status );
	unitAttr.setKeyable(true);
	unitAttr.setAffectsWorldSpace(true);
	unitAttr.setStorable(true);
	return status;
}

MStatus AHelper::createTimeAttrInternal(MObject& attr, const MString& nameLong,const MString& nameShort, double val)
{
	MStatus status;
	MFnUnitAttribute unitAttr;
	attr = unitAttr.create(nameLong, nameShort, MFnUnitAttribute::kTime, val, &status );
	unitAttr.setKeyable(true);
	unitAttr.setAffectsWorldSpace(true);
	unitAttr.setStorable(true);
	unitAttr.setInternal(true);
	return status;
}

MMatrix AHelper::getMatrixAttr(const MObject& node, MObject& attr)
{
	MPlug matplg( node, attr );
	MObject matobj;
	matplg.getValue(matobj);
	MFnMatrixData matdata(matobj);
    	return matdata.matrix();
}

MString AHelper::getStringAttr(const MObject& node, MObject& attr)
{
	MPlug strplg( node, attr );
	MString res;
	strplg.getValue(res);
    	return res;
}

MStatus AHelper::createDoubleAttr(MObject& attr, const MString& nameLong, const MString& nameShort, double val)
{
	MStatus status;
	MFnNumericAttribute fAttr;
	attr = fAttr.create(nameLong, nameShort,  MFnNumericData::kDouble, val, &status);
	fAttr.setStorable(true);
	fAttr.setKeyable(true);
	fAttr.setReadable(true);
	fAttr.setConnectable(true);
	return status;
}

#include <string>

void AHelper::validateFilePath(MString& name)
{
	std::string str(name.asChar());

	int found = str.find('|', 0);
	
	while(found>-1)
	{
		str[found] = '_';
		found = str.find('|', found);
	}
	
	found = str.find(':', 0);
	
	while(found>-1)
	{
		str[found] = '_';
		found = str.find(':', found);
	}
		
		name = MString(str.c_str());
}

void AHelper::noDotDagPath(MString& name)
{
	std::string str(name.asChar());

	int found = str.find('.', 0);
	
	while(found>-1)
	{
		str[found] = '_';
		found = str.find('.', found);
	}
		
		name = MString(str.c_str());
}

void AHelper::getWindowsPath(MString& name)
{
	std::string str(name.asChar());

	int found = str.find('/', 0);
	
	while(found>-1)
	{
		str[found] = '\\';
		found = str.find('/', found);
	}
		
		name = MString(str.c_str());
}

void AHelper::getFileNameFirstDot(MString& name)
{
	std::string str(name.asChar());

	int found = str.find('.', 0);
	if(found>-1)str.erase(found);
		
	name = MString(str.c_str());
}

void AHelper::cutFileNameLastSlash(MString& name)
{
	std::string str(name.asChar());

	int found = str.find_last_of('/', str.size()-1);
	if(found>-1)str.erase(0, found+1);
		
	name = MString(str.c_str());
}

void AHelper::changeFileNameExtension(MString& name, const char* ext)
{
	std::string str(name.asChar());

	int found = str.find_last_of('.', str.size()-1);
	if(found>-1)str.erase(found);
		
	name = MString(str.c_str());

	name += MString(ext);
}

void AHelper::getProjectDataPath(MString& path)
{
	MGlobal::executeCommand(MString ("string $p = `workspace -q -fn`"), path);
	path = path+"/data/";
}

int AHelper::hasNamedAttribute(const MObject& node, const char* attrname)
{
	MFnDependencyNode fnode(node);
			
	MStatus res;
	fnode.attribute(attrname, &res);
	if(res == MS::kFailure) return 0;
		
	return 1;
}

void AHelper::displayIntParam(const char* str, int val)
{
	MGlobal::displayInfo(MString(str) + ": " + val);
}

void AHelper::displayVectorParam(const char* str, double x, double y, double z)
{
	MGlobal::displayInfo(MString(str) + ": " + x + " " + y + " " + z);
}

void AHelper::getColorAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& r, double& g, double& b)
{
	MPlug plgR = fnode.findPlug(MString(attrname)+"R");
	plgR.getValue(r);
	
	MPlug plgG = fnode.findPlug(MString(attrname)+"G");
	plgG.getValue(g);
	
	MPlug plgB = fnode.findPlug(MString(attrname)+"B");
	plgB.getValue(b);
}

void AHelper::getNormalAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& r, double& g, double& b)
{
	MPlug plgR = fnode.findPlug(MString(attrname)+"X");
	plgR.getValue(r);
	
	MPlug plgG = fnode.findPlug(MString(attrname)+"Y");
	plgG.getValue(g);
	
	MPlug plgB = fnode.findPlug(MString(attrname)+"Z");
	plgB.getValue(b);
}

char AHelper::getBoolAttributeByName(const MFnDependencyNode& fnode, const char* attrname, bool& v)
{
	MPlug plgV = fnode.findPlug(MString(attrname));
	if(plgV.isNull()) return 0;
	plgV.getValue(v);
	return 1;
}

char AHelper::getDoubleAttributeByName(const MFnDependencyNode& fnode, const char* attrname, double& v)
{
	MPlug plgV = fnode.findPlug(MString(attrname));
	if(plgV.isNull()) return 0;
	plgV.getValue(v);
	return 1;
}

char AHelper::getDoubleAttributeByNameAndTime(const MFnDependencyNode& fnode, const char* attrname, MDGContext & ctx, double& v)
{
	MPlug plgV = fnode.findPlug(MString(attrname));
	if(plgV.isNull()) return 0;
	plgV.getValue(v, ctx);
	return 1;
}

char AHelper::getStringAttributeByName(const MFnDependencyNode& fnode, const char* attrname, MString& v)
{
	MPlug plgV = fnode.findPlug(MString(attrname));
	if(plgV.isNull()) return 0;
	plgV.getValue(v);
	return 1;
}

char AHelper::getStringAttributeByName(const MObject& node, const char* attrname, MString& v)
{
	MFnDependencyNode fnode(node);
	MPlug plgV = fnode.findPlug(MString(attrname));
	if(plgV.isNull()) return 0;
	plgV.getValue(v);
	return 1;
}

int AHelper::getConnectedAttributeByName(const MFnDependencyNode& fnode, const char* attrname, MString& v)
{
	MPlug plgTo = fnode.findPlug(MString(attrname));
	
	if(!plgTo.isConnected()) return 0;
	
	MPlugArray conns;
	
	plgTo.connectedTo (conns, true, false);
	
	v = conns[0].name();
	
	return 1;
}

void AHelper::getMat(const MMatrix &mat, float space[16])
{
	space[0] = mat[0][0]; 
	space[1] = mat[0][1]; 
	space[2] = mat[0][2];
	space[3] = mat[0][3];
	space[4] = mat[1][0]; 
	space[5] = mat[1][1]; 
	space[6] = mat[1][2];
	space[7] = mat[1][3];
	space[8] = mat[2][0]; 
	space[9] = mat[2][1]; 
	space[10] = mat[2][2];
	space[11] = mat[2][3];
	space[12] = mat[3][0]; 
	space[13] = mat[3][1]; 
	space[14] = mat[3][2];
	space[15] = mat[3][3];
}

void AHelper::getTransformMat(const MDagPath &path, float space[16])
{
	MFnTransform ftransform(path); 
	MMatrix mat = ftransform.transformation().asMatrix();
	space[0]  = mat[0][0]; 
	space[1]  = mat[0][1]; 
	space[2]  = mat[0][2];
	space[3]  = mat[0][3];
	space[4]  = mat[1][0]; 
	space[5]  = mat[1][1]; 
	space[6]  = mat[1][2];
	space[7]  = mat[1][3];
	space[8]  = mat[2][0]; 
	space[9]  = mat[2][1]; 
	space[10] = mat[2][2];
	space[11] = mat[2][3];
	space[12] = mat[3][0]; 
	space[13] = mat[3][1]; 
	space[14] = mat[3][2];
	space[15] = mat[3][3];
}

void AHelper::getTransformWorld(const MString& name, float space[4][4])
{
	MDoubleArray fm;
	MGlobal::executeCommand(MString("xform -q -m -ws ")+name,fm );
	space[0][0]=fm[0]; space[0][1]=fm[1]; space[0][2]=fm[2];
	space[1][0]=fm[4]; space[1][1]=fm[5]; space[1][2]=fm[6];
	space[2][0]=fm[8]; space[2][1]=fm[9]; space[2][2]=fm[10];
	space[3][0]=fm[12]; space[3][1]=fm[13]; space[3][2]=fm[14];
	fm.clear();
}

MVector AHelper::getTransformWorldNoScale(const MString& name, float space[4][4])
{
	MDoubleArray fm;
	MGlobal::executeCommand(MString("xform -q -m -ws ")+name,fm );
	MVector scale;
	MVector vx(fm[0], fm[1], fm[2]); scale.x = vx.length(); vx.normalize();
	MVector vy(fm[4], fm[5], fm[6]); scale.y = vy.length(); vy.normalize();
	MVector vz(fm[8], fm[9], fm[10]); scale.z = vz.length(); vz.normalize();
	space[0][0]=vx.x; space[0][1]=vx.y; space[0][2]=vx.z;
	space[1][0]=vy.x; space[1][1]=vy.y; space[1][2]=vy.z;
	space[2][0]=vz.x; space[2][1]=vz.y; space[2][2]=vz.z;
	space[3][0]=fm[12]; space[3][1]=fm[13]; space[3][2]=fm[14];
	fm.clear();
	return scale;
}

MTime::Unit AHelper::getPlaybackFPS()
{
	MString rfps;
	MGlobal::executeCommand(MString("currentUnit -q -time"), rfps, 0, 0);
	
	MTime::Unit tu; 
	
	if(rfps == "film") 
	{
		tu = MTime::kFilm;
	}
	else if(rfps == "pal") 
	{
		tu = MTime::kPALFrame;
	}
	else 
	{
		tu = MTime::kNTSCFrame;
	}
	
	return tu;
}


MObject AHelper::createMinimalMesh(MObject& parent, const char* name)
{
	MPointArray vertexArray;
	vertexArray.append(MPoint(0.0, 0.0, 0.0));
	vertexArray.append(MPoint(1.0, 0.0, 0.0));
	vertexArray.append(MPoint(1.0, 0.0, 1.0));

	MIntArray polygonCounts;
	polygonCounts.append(3);
	
	MIntArray polygonConnects;
	polygonConnects.append(0);
	polygonConnects.append(1);
	polygonConnects.append(2);

	MFnMesh fmesh;
	MObject amesh = fmesh.create( 3, 1, vertexArray, polygonCounts, polygonConnects, parent);
	fmesh.setName( name );
	
	return amesh;
}

char AHelper::findNamedPlugsInHistory(MObject &root, MFn::Type type, MString &name1, MPlug &plug1, MString &name2, MPlug &plug2)
{
        MStatus stat;
	MItDependencyGraph itdep(root, type, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next())
	{
	        MObject ocur = itdep.currentItem();
	        MFnDependencyNode fdep(ocur);
	        plug1 = fdep.findPlug( name1, false, &stat);
	        plug2 = fdep.findPlug( name2, false, &stat);
	        if(stat)
	                return 1;
	}
	return 0;
}

void AHelper::convertToMatrix(MMatrix& mat, float mm[4][4])
{
	mm[0][0] = mat.matrix[0][0];
	mm[0][1] = mat.matrix[0][1];
	mm[0][2] = mat.matrix[0][2];
	mm[0][3] = 0.f;
	mm[1][0] = mat.matrix[1][0];
	mm[1][1] = mat.matrix[1][1];
	mm[1][2] = mat.matrix[1][2];
	mm[1][3] = 0.f;
	mm[2][0] = mat.matrix[2][0];
	mm[2][1] = mat.matrix[2][1];
	mm[2][2] = mat.matrix[2][2];
	mm[2][3] = 0.f;
	mm[3][0] = mat.matrix[3][0];
	mm[3][1] = mat.matrix[3][1];
	mm[3][2] = mat.matrix[3][2];
	mm[3][3] = 1.f;
}

void AHelper::convertToMatrix(float *m, float mm[4][4])
{
	mm[0][0] = m[0];
	mm[0][1] = m[1];
	mm[0][2] = m[2];
	mm[0][3] = m[3];
	mm[1][0] = m[4];
	mm[1][1] = m[5];
	mm[1][2] = m[6];
	mm[1][3] = m[7];
	mm[2][0] = m[8];
	mm[2][1] = m[9];
	mm[2][2] = m[10];
	mm[2][3] = m[11];
	mm[3][0] = m[12];
	mm[3][1] = m[13];
	mm[3][2] = m[14];
	mm[3][3] = m[15];
}

char AHelper::findNamedDagInList(MDagPathArray &arr, const char *name)
{
	int narr = arr.length();
	for(int i = 0; i < narr; i++)
	{
		if(arr[i].fullPathName() == name)
			return 1;
	}
	return 0;
}

void AHelper::createMinimalMaterial(MString &name, MString &texture)
{
	MGlobal::executeCommand(MString("shadingNode -asShader lambert -name ") + name);
	MGlobal::executeCommand(MString("sets -renderable true -noSurfaceShader true -empty -name ") + name + "SG");
	MGlobal::executeCommand(MString("connectAttr -f ") + name + ".outColor " + name + "SG.surfaceShader");
	if(texture != "")
	{
		MGlobal::executeCommand(MString("shadingNode -asTexture file -name ") + name + "_color_file");
		MGlobal::executeCommand(MString("connectAttr -force ") + name + "_color_file.outColor " + name + ".color");
		MGlobal::executeCommand(MString("setAttr ")+ name + ("_color_file.fileTextureName -type \"string\" \"") + texture + "\"");
	}
}

char AHelper::containsGeom(const MDagPath & root)
{
	MItDag itdag(MItDag::kBreadthFirst);
	itdag.reset(root);
	for(; !itdag.isDone(); itdag.next()) {
		MObject cur = itdag.currentItem();
		if(cur.hasFn(MFn::kMesh) || cur.hasFn(MFn::kNurbsCurve) || cur.hasFn(MFn::kCamera))
			return 1;
	}
	return 0;
}

std::string AHelper::FullPathNameToObj(const MObject & node)
{
    MFnDagNode pf(node);
    return std::string(pf.fullPathName().asChar());
}

MMatrix AHelper::GetWorldTransformMatrix(const MDagPath & path)
{
	MMatrix m = MMatrix::identity;
	if(path.node().hasFn(MFn::kTransform))
		m = MFnTransform(path).transformation().asMatrix();
		
	m *= GetWorldParentTransformMatrix(path);
	return m;
}

MMatrix AHelper::GetWorldParentTransformMatrix(const MDagPath & path)
{
	MMatrix m;
    MDagPath parentPath = path;
    MStatus stat;
    for(;;) {
        stat = parentPath.pop();
        if(!stat) break;
        MFnTransform ft(parentPath, &stat);
        if(!stat) break;   

        m *= ft.transformation().asMatrix();
    }
    return m;
}

void AHelper::PrintMatrix(const std::string & note, const MMatrix & mat)
{
    std::stringstream sst;
	sst<<note<<" ["<<mat[0][0]<<", "<<mat[0][1]<<", "<<mat[0][2]<<", "<<mat[0][3]<<"]\n"
    <<"["<<mat[1][0]<<", "<<mat[1][1]<<", "<<mat[1][2]<<", "<<mat[1][3]<<"]\n"
    <<"["<<mat[2][0]<<", "<<mat[2][1]<<", "<<mat[2][2]<<", "<<mat[2][3]<<"]\n"
    <<"["<<mat[3][0]<<", "<<mat[3][1]<<", "<<mat[3][2]<<", "<<mat[3][3]<<"]\n";
	MGlobal::displayInfo(sst.str().c_str());
}

void AHelper::ConvertToMMatrix(MMatrix & dst, const Matrix44F & src)
{
    dst[0][0] = src.M(0,0);
    dst[0][1] = src.M(0,1);
    dst[0][2] = src.M(0,2);
    dst[0][3] = src.M(0,3);
    dst[1][0] = src.M(1,0);
    dst[1][1] = src.M(1,1);
    dst[1][2] = src.M(1,2);
    dst[1][3] = src.M(1,3);
    dst[2][0] = src.M(2,0);
    dst[2][1] = src.M(2,1);
    dst[2][2] = src.M(2,2);
    dst[2][3] = src.M(2,3);
    dst[3][0] = src.M(3,0);
    dst[3][1] = src.M(3,1);
    dst[3][2] = src.M(3,2);
    dst[3][3] = src.M(3,3);
}

void AHelper::ConvertToMatrix44F(Matrix44F & dst, const MMatrix & src)
{
    *dst.m(0, 0) = src.matrix[0][0];
    *dst.m(0, 1) = src.matrix[0][1];
    *dst.m(0, 2) = src.matrix[0][2];
    *dst.m(0, 3) = src.matrix[0][3];
    *dst.m(1, 0) = src.matrix[1][0];
    *dst.m(1, 1) = src.matrix[1][1];
    *dst.m(1, 2) = src.matrix[1][2];
    *dst.m(1, 3) = src.matrix[1][3];
    *dst.m(2, 0) = src.matrix[2][0];
    *dst.m(2, 1) = src.matrix[2][1];
    *dst.m(2, 2) = src.matrix[2][2];
    *dst.m(2, 3) = src.matrix[2][3];
    *dst.m(3, 0) = src.matrix[3][0];
    *dst.m(3, 1) = src.matrix[3][1];
    *dst.m(3, 2) = src.matrix[3][2];
    *dst.m(3, 3) = src.matrix[3][3];
}

void AHelper::SimpleAnimation(const MPlug & dst, int a, int b)
{
	float dQ = b - a;
	float dt = b - a;
	float ang = atan(dQ/dt);
	ang = AngleToDegree<float>(a);
	
	float length = sqrt(dQ*dQ + dt*dt) * 0.33;
	
	float inAngle = ang;
	float inWeight = length;
		
	float outAngle = ang;
	float outWeight = length;
	
	MAngle angle(inAngle, MAngle::kDegrees);
	MAngle angleo(outAngle, MAngle::kDegrees);

	MFnAnimCurve animCv;
	animCv.create ( dst, MFnAnimCurve::kAnimCurveTU );
	animCv.setIsWeighted(true);
	MTime::Unit timeUnit = MTime::uiUnit();
	
	MTime tmt(a, timeUnit);
	double v = a;
	
	animCv.addKeyframe(tmt, v);
	animCv.setTangentsLocked(0, false);
	animCv.setWeightsLocked(0, false);
	animCv.setTangent(0, angle, inWeight, true);
	animCv.setTangent(0, angleo, outWeight, false);
	animCv.setInTangentType(0, MFnAnimCurve::kTangentLinear );
	animCv.setOutTangentType(0, MFnAnimCurve::kTangentLinear );
	
	tmt = MTime(b, timeUnit);
	v = b;
	
	animCv.addKeyframe(tmt, v);
	animCv.setTangentsLocked(1, false);
	animCv.setWeightsLocked(1, false);
	animCv.setTangent(1, angle, inWeight, true);
	animCv.setTangent(1, angleo, outWeight, false);
	animCv.setInTangentType(1, MFnAnimCurve::kTangentLinear );
	animCv.setOutTangentType(1, MFnAnimCurve::kTangentLinear );
}

bool AHelper::IsReferenced(const MObject & node)
{
    MFnDependencyNode fnode(node );
    return fnode.isFromReferencedFile();
}

MObject AHelper::CreateDeformer(const MString & name)
{
	MString cmd = MString("deformer -type ") + name;
	MStringArray result;
	MGlobal::executeCommand ( cmd, result);
	if(result.length() < 1) {
		AHelper::Info<MString>("AHelper error cannot create deformer", name);
		return MObject::kNullObj;
	}
	
	AHelper::Info<MString>("AHelper create deformer", result[0]);
	
	MGlobal::selectByName (result[0], MGlobal::kReplaceList );
	MSelectionList sels;
	MGlobal::getActiveSelectionList ( sels );
	
	MItSelectionList iter( sels );
	MObject node;
	iter.getDependNode(node);
	return node;
}
//:~