/*
 *  aSearchHelper.cpp
 *  opium
 *
 *  Created by jian zhang on 6/14/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */
#include <maya/MFnSet.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MFnDagNode.h>
#include <maya/MItDag.h>
#include <maya/MItDependencyGraph.h>
#include <maya/MItDependencyNodes.h>
#include <maya/MMatrix.h>
#include <maya/MFnTransform.h>
#include <maya/MFnMesh.h>
#include <maya/MSelectionList.h>
#include <maya/MItSelectionList.h>
#include <maya/MGlobal.h>
#include "ASearchHelper.h"
#include <SHelper.h>

#include <boost/foreach.hpp>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/format.hpp>

#include <string>
#include <vector>
using namespace std;

std::string ASearchHelper::getPullPathName(MObject &node)
{
    MFnDagNode pf(node);
    return std::string(pf.fullPathName().asChar());
}

char ASearchHelper::findObjDirectChildByName(MObject &parent, MObject &result, std::string &name)
{
	//SHelper::stripAll(name);
	
	if(parent != MObject::kNullObj)
	{
		MFnDagNode ppf(parent);
		for(unsigned i = 0; i <ppf.childCount(); i++)
		{
			MFnDagNode pf(ppf.child(i));
			std::string curname = pf.name().asChar();
			//SHelper::stripAll(curname);
			if(curname == name) 
			{
				result = ppf.child(i);
				return 1;
			}
		}
	}
	else
	{
		MItDag itdag(MItDag::kBreadthFirst);
		for(; !itdag.isDone(); itdag.next())
		{
			MFnDagNode pf(itdag.currentItem());
			std::string curname = pf.name().asChar();
			//SHelper::stripAll(curname);
			if(curname == name) 
			{
				result = itdag.currentItem();
				return 1;
			}
		}
	}
	
	result = MObject::kNullObj;
	return 0;
}

char ASearchHelper::findObjDirectChildByNameIgnoreNamespace(MObject &parent,MObject &result,std::string &name)
{
	if(parent!=MObject::kNullObj) {
		MFnDagNode ppf(parent);
		for(unsigned i=0; i < ppf.childCount(); i++)
		{
			MFnDagNode pf(ppf.child(i));
			std::string curname = pf.name().asChar();
			curname = SHelper::removeNamespace(curname);
			if(name == curname)
			{
				result = ppf.child(i);
				return 1;
			}
		}
	}
	else {
		MItDag itdag(MItDag::kBreadthFirst);
		for(; !itdag.isDone(); itdag.next())
		{
			MFnDagNode pf(itdag.currentItem());
			std::string curname = pf.name().asChar();
			curname = SHelper::removeNamespace(curname);
			if(name == curname) {
				result = itdag.currentItem();
				return 1;
			}
		}
	}
	result = MObject::kNullObj;
	return 0;
}

char ASearchHelper::findObjWithNamespaceDirectChildByName(MObject &parent,MObject &result,std::string &name)
{
	if(parent!=MObject::kNullObj)
	{
		MFnDagNode ppf(parent);
		for(unsigned i=0;i<ppf.childCount();i++)
		{
			MFnDagNode pf(ppf.child(i));
			std::string curname = pf.name().asChar();
		    if(SHelper::fuzzyMatchNamespace(name,curname))
			  {
				result = ppf.child(i);
				return 1;
			  }
	
		}
	}

	else
	{
		MItDag itdag(MItDag::kBreadthFirst);
		for(; !itdag.isDone(); itdag.next())
		{
			MFnDagNode pf(itdag.currentItem());
			std::string curname = pf.name().asChar();
			
			 if(SHelper::fuzzyMatchNamespace(name,curname)) 
			  {
				result = itdag.currentItem();
				return 1;
			  }
			
		}
	}
	result = MObject::kNullObj;
	return 0;
}


char ASearchHelper::getObjByFullName(const char* name, MObject& res, MObject& root)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(str, sep);
	MObject parent = root;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		if(!findObjDirectChildByName( parent, res, r))
			return 0;
		
		parent = res;
	}
	return 1;
}

char ASearchHelper::getObjByFullNameIgnoreNamespace(const char* name, MObject& res, MObject& root)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	MObject parent = root;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		r = SHelper::removeNamespace(r);
		if(!findObjDirectChildByNameIgnoreNamespace( parent, res, r))
			return 0;
		
		parent = res;
	}
	return 1;
}

char ASearchHelper::fuzzyGetObjByFullName(std::string &toReplace,const char* name, MObject& res, MObject& root)
{
	std::string r;
	std::string str = name;
	SHelper::replaceAnyNamespace(str,toReplace);
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	MObject parent = root;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		if(!findObjWithNamespaceDirectChildByName( parent, res, r))
			return 0;
		
		parent = res;
	}
	return 1;
}

char ASearchHelper::getDescendedByFullNameIgnoreNamespace(MObject& root, const char* name, MObject& res)
{
	std::string r;
	std::string str = name;
	SHelper::removeAnyNamespace(str);
	
	std::string rootName = getPullPathName(root);
	SHelper::removeAnyNamespace(rootName);
	
	if(boost::algorithm::starts_with(str, rootName)) 
	{
	    boost::algorithm::replace_first(str, rootName, "");
	}
	
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	MObject parent = root;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		if(!findObjDirectChildByNameIgnoreNamespace( parent, res, r))
		{
			return 0;
		}
		
		parent = res;
	}
	return 1;
}

char ASearchHelper::fuzzyGetDescendedByTerminalName(MObject& root, const char* name, MObject& res)
{
    MItDag itdag(MItDag::kBreadthFirst);
    if(root != MObject::kNullObj)
        itdag.reset (root);
    for(; !itdag.isDone(); itdag.next()) {
        MFnDagNode pf(itdag.currentItem());
        std::string curname = pf.name().asChar();
        if(SHelper::isMatched(name, curname)) {
            res = itdag.currentItem();
            return 1;
        }
    }
    return 0;
}

char ASearchHelper::findFirstTypedChild(MDagPath &parent, MObject &result, MFn::Type type)
{
    MItDag itdag(MItDag::kBreadthFirst);
    itdag.reset(parent);
    for(; !itdag.isDone(); itdag.next()) {
        result = itdag.currentItem();
        if(result.hasFn(type)) {
            return 1;
        }
    }
    return 0;
}

char ASearchHelper::findTypedNodeInHistory(MObject &root, const char *nodename, MObject &res, bool downstream)
{
	if(root == MObject::kNullObj) return 0;
	MItDependencyGraph::Direction dir = downstream ? MItDependencyGraph::kDownstream : MItDependencyGraph::kUpstream;
	MStatus stat;
	MItDependencyGraph itdep(root, MFn::kInvalid, dir, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next()) {
		if(MFnDependencyNode(itdep.currentItem()).typeName() == nodename) {
			res = itdep.currentItem();
			return 1;
		}
	}
	return 0;
}

char ASearchHelper::findNamedPlugInHistory(MObject &root, MFn::Type type, MString &name1, MPlug &plug1)
{
	MStatus stat;
	MItDependencyGraph itdep(root, type, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next())
	{
	        MObject ocur = itdep.currentItem();
	        MFnDependencyNode fdep(ocur);
	        plug1 = fdep.findPlug( name1, false, &stat);
	        if(stat)
	                return 1;
	}
	return 0;
}

char ASearchHelper::isObjInDownstream(MObject &root, MObject &obj)
{
    MStatus stat;
	MItDependencyGraph itdep(root, MFn::kInvalid, MItDependencyGraph::kDownstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next())
	{
	        MObject ocur = itdep.currentItem();
	        if(ocur == obj)
                return 1;
	}
	return 0;
}

char ASearchHelper::getWorldTMByFullName(const char* name, MMatrix& res)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	MObject parent = MObject::kNullObj;
	MObject child;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		if(!findObjDirectChildByName( parent, child, r))
			return 0;
			
		MMatrix curtm = MFnTransform(child).transformation().asMatrix();
		res *= curtm;
		
		parent = child;
	}
	return 1;
}

char ASearchHelper::getWorldTMByObj(const MObject& root, MMatrix &res)
{	
	MString name = MFnDagNode(root).fullPathName();
	return getWorldTMByFullName(name.asChar(), res);
}

char ASearchHelper::isFirstNonIdentityTM(const MObject& root, MMatrix &res)
{
	res = MFnTransform(root).transformation().asMatrix();
	if(res == MMatrix::identity)
		return 0;
	
	std::string r;
	std::string str = MFnDagNode(root).fullPathName().asChar();
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	MObject parent = MObject::kNullObj;
	MObject child;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		if(!findObjDirectChildByName( parent, child, r))
			return 0;

		tokenizer::iterator queue = tok_iter;
		queue++;
		if(queue != tokens.end())
		{
			MMatrix curtm = MFnTransform(child).transformation().asMatrix();
			if(curtm != MMatrix::identity)
				return 0;
		}
		
		parent = child;
	}
	return 1;
}

char ASearchHelper::isParentAllIdentityTM(const MDagPath &path)
{
	std::string r;
	std::string str = MFnDagNode(path).fullPathName().asChar();
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|");
	tokenizer tokens(str, sep);
	MObject parent = MObject::kNullObj;
	MObject child;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		if(!findObjDirectChildByName( parent, child, r))
			return 0;

		tokenizer::iterator queue = tok_iter;
		queue++;
		if(queue != tokens.end())
		{
			MMatrix curtm = MFnTransform(child).transformation().asMatrix();
			if(curtm != MMatrix::identity)
				return 0;
		}
		
		parent = child;
	}

	return 1;
}	

MObject ASearchHelper::findShadingEngine(MObject &mesh)
{
	MStatus result;
	MItDependencyGraph itdep(mesh, MFn::kShadingEngine, MItDependencyGraph::kDownstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &result);
	
	for(; !itdep.isDone(); itdep.next())
	{
		return itdep.currentItem();
	}
	
	return MObject::kNullObj;
}

MObject ASearchHelper::findMaterial(MObject &mesh)
{
	MObject osg = findShadingEngine(mesh);
	if(osg == MObject::kNullObj)
		return osg;
	
	MStatus result;
	MPlug plgsurface = MFnDependencyNode(osg).findPlug("surfaceShader");
	
	MItDependencyGraph itdep(plgsurface, MFn::kInvalid, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &result);
	
	for(; !itdep.isDone(); itdep.next())
	{
		MObject ocur = itdep.currentItem();
		if(ocur.hasFn(MFn::kLambert) || ocur.hasFn(MFn::kBlinn) || ocur.hasFn(MFn::kPhong))
			return ocur;
	}
	
	return MObject::kNullObj;
}

char ASearchHelper::getFiles(MObject &root, MObjectArray &arr)
{
    if(root.hasFn(MFn::kFileTexture))
    {
        if(!isObjInArray(root, arr))
            arr.append(root);
    }
	MStatus result;
	MItDependencyGraph itdep(root, MFn::kFileTexture, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &result);
	
	for(; !itdep.isDone(); itdep.next())
	{
		MObject ocur = itdep.currentItem();
		if(!isObjInArray(ocur, arr))
		    arr.append(ocur);
	}
	return 1;
}

char ASearchHelper::allTypedNodeInHistory(MObject &root, MObjectArray &arr, const char *typeName)
{
    MStatus result;
    MItDependencyGraph itdep(root, MFn::kInvalid, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &result);
	
	for(; !itdep.isDone(); itdep.next())
	{
		MObject ocur = itdep.currentItem();
		if(MFnDependencyNode(ocur).typeName() == typeName)
		{
		    if(!isObjInArray(ocur, arr))
		        arr.append(ocur);
		}
	}
	return 1;
}

char ASearchHelper::isStringInArray(const std::vector<std::string> &arr, const std::string &val)
{
	BOOST_FOREACH(std::string a, arr)
    {
        if( a == val) return 1;
    }
	return 0;
}

char ASearchHelper::isObjInArray(MObject &obj, MObjectArray &arr)
{
    for(unsigned i=0; i < arr.length(); i++)
    {
        if(obj == arr[i])
            return 1;
    }
    return 0;
}

char ASearchHelper::isPathInArray(MDagPath &path, MDagPathArray &arr)
{
	for(unsigned i=0; i < arr.length(); i++)
    {
        if(path == arr[i])
            return 1;
    }
    return 0;
}

char ASearchHelper::findIntermediateMeshInHistory(MObject &root, MObject &res)
{
	if(root == MObject::kNullObj)
		return 0;
		
	MStatus stat;
	MItDependencyGraph itdep(root, MFn::kMesh, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next())
	{
		if(MFnDagNode(itdep.currentItem()).isIntermediateObject())
		{
			res = itdep.currentItem();
			return 1;
		}
	}
	return 0;
}

char ASearchHelper::findMatchedMeshWithNamedPlugsInHistory(MObject &root, MObject &res, MString &name1, MPlug &plug1, MString &name2, MPlug &plug2)
{
	MStatus stat;
	MFnMesh fmesh(root);
	int should_have = fmesh.numVertices();
	MItDependencyGraph itdep(root, MFn::kMesh, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next())
	{
		MObject ocur = itdep.currentItem();
		MFnDependencyNode fdep(ocur);
		plug1 = fdep.findPlug( name1, false, &stat);
		plug2 = fdep.findPlug( name2, false, &stat);
		if(stat) // has cache history
		{
			MFnMesh fmatch(ocur);
			if(fmatch.numVertices() == should_have) // and match num vert
			{
				res = ocur;
				return 1;
			}
		}
	}
	res = MObject::kNullObj;
	return 0;
}

char ASearchHelper::findReferencedMeshInHistory(MObject &root, MObject &res)
{
    MStatus stat;
	MItDependencyGraph itdep(root, MFn::kMesh, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next()) {
		MObject ocur = itdep.currentItem();
		MFnDependencyNode fdep(ocur);
		MFnDagNode fdag(ocur);
		if(fdep.isFromReferencedFile()) {
		    res = ocur;
			return 1;
		}
	}
	res = MObject::kNullObj;
	return 0;
}

char ASearchHelper::findTransformWithNamedPlugValue(const MObject &root, MObject &res, MString &name, MString &value)
{
	MStatus stat;
	MItDag itdag(MItDag::kBreadthFirst);
	itdag.reset(root);
	for(; !itdag.isDone(); itdag.next())
	{
		MObject ocur = itdag.currentItem();
		MFnDependencyNode fdep(ocur);
		MPlug plg = fdep.findPlug( name, false, &stat);
		if(stat) // has cache history
		{
			MString valToMatch;
			plg.getValue(valToMatch);
			if(valToMatch == value)
			{
				res = ocur;
				return 1;
			}
		}
	}
	
	res = MObject::kNullObj;
	return 0;
}

char ASearchHelper::findNodeWithNamedPlugValue2(const MObject &root, MObject &res, MString &name1, MString &value1, MString &name2, MString &value2)
{
	MStatus stat;
	MItDag itdag(MItDag::kBreadthFirst);
	itdag.reset(root);
	for(; !itdag.isDone(); itdag.next())
	{
		MObject ocur = itdag.currentItem();
		MFnDependencyNode fdep(ocur);
		MPlug plg1 = fdep.findPlug( name1, false, &stat);
		MPlug plg2 = fdep.findPlug( name2, false, &stat);
		if(stat) // has cache history
		{
			MString valToMatch1;
			plg1.getValue(valToMatch1);
			MString valToMatch2;
			plg2.getValue(valToMatch2);
			if(valToMatch1 == value1 && valToMatch2 == value2)
			{
				res = ocur;
				return 1;
			}
		}
	}
	
	res = MObject::kNullObj;
	return 0;
}

bool ASearchHelper::FirstConnectedTypedDepNodeByTypename(MFn::Type type, MString& name, MObject& root, MObject& node)
{
	node = MObject::kNullObj;
	MStatus stat;
	// up stream
	MItDependencyGraph itdep(root, type, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next()) {
		MFnDependencyNode pf(itdep.currentItem());
		if(pf.typeName()==name) {
			node = itdep.currentItem();
			return true;
		}
	} 
	// down stream
	MItDependencyGraph itdepd(root, type, MItDependencyGraph::kDownstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdepd.isDone(); itdepd.next()) {
		MFnDependencyNode pf(itdepd.currentItem());
		if(pf.typeName()==name) {
			node = itdepd.currentItem();
			return true;
		}
	}
	return false;
}

char ASearchHelper::findShadingEnginesByName(std::string& name, MObjectArray& result)
{
	name = SHelper::removeNamespace(name);
	char found = 0;
	MItDependencyNodes itdep(MFn::kShadingEngine);
	for(; !itdep.isDone(); itdep.next())
	{
		MObject ocur = itdep.thisNode();
		std::string curname(MFnDependencyNode(ocur).name().asChar());
		curname = SHelper::removeNamespace(curname);
		if(name == curname)
		{
			found = 1;
			result.append(ocur);
		}
	}
	return found;
}

char ASearchHelper::findSkinByMesh(MObject& mesh, MObject &skin)
{
	skin = MObject::kNullObj;
	MStatus stat;
	MItDependencyGraph itdep(mesh, MFn::kSkinClusterFilter, MItDependencyGraph::kUpstream , MItDependencyGraph::kDepthFirst , MItDependencyGraph::kNodeLevel, &stat);
	for(; !itdep.isDone(); itdep.next())
	{
		skin = itdep.thisNode();
	}
	return skin != MObject::kNullObj;
}

char ASearchHelper::shadingEngines(MObject &mesh, MObjectArray & sgs)
{
	MStatus result;
	MItDependencyGraph itdep(mesh, MFn::kShadingEngine, MItDependencyGraph::kDownstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &result);
	sgs.clear();
	for(; !itdep.isDone(); itdep.next())
	{
		sgs.append(itdep.currentItem());
	}
	
	return sgs.length() > 0;
}

char ASearchHelper::facesAttachedToShadingEngine(MObject &mesh, MObject & sg, MDagPath & item, MObject & component)
{
	MStatus stat;
	MPlug sets = MFnDependencyNode(sg).findPlug("dagSetMembers");
	MSelectionList allfaces;
	for(int i = 0; i < sets.numConnectedElements(); i++)
	{
		MPlug start = sets.elementByLogicalIndex(i);
		MItDependencyGraph itdep(start, MFn::kMesh, MItDependencyGraph::kUpstream , MItDependencyGraph::kDepthFirst , MItDependencyGraph::kNodeLevel, &stat);
		for(; !itdep.isDone(); itdep.next())
		{
			MPlug objectGroups = itdep.thisPlug();
			if(objectGroups.node() == mesh)
			{
				MGlobal::displayInfo(start.name() + " <-- " + objectGroups.name());
			
				MDGModifier mdif;
				MObject faset = mdif.createNode("objectSet");
				mdif.doIt ();
	
				MPlug dst = MFnDependencyNode(faset).findPlug("dagSetMembers").elementByLogicalIndex(0);
				mdif.connect(objectGroups, dst );
				mdif.doIt ();
				
				MFnSet fset(faset, &stat);
				
				MSelectionList selfaces;
				fset.getMembers (selfaces, 0);
				allfaces.merge(selfaces, MSelectionList::kMergeNormal);

				mdif.deleteNode(faset);
				mdif.doIt();
			}
		}
	}
	
	MItSelectionList itsel(allfaces);
	for ( ; !itsel.isDone(); itsel.next() )
	{
		
		itsel.getDagPath( item, component );
	}
	return 1;
}

char ASearchHelper::dagDirectChildByName(MDagPath &parent, MDagPath &result, std::string &name)
{
	if(parent.isValid())
	{
		MFnDagNode ppf(parent);
		for(unsigned i = 0; i <ppf.childCount(); i++)
		{
			MFnDagNode pf(ppf.child(i));
			std::string curname = pf.name().asChar();
			//SHelper::stripAll(curname);
			if(curname == name) 
			{
				MFnDagNode(ppf.child(i)).getPath(result);
				return 1;
			}
		}
	}
	else
	{
		MItDag itdag(MItDag::kBreadthFirst);
		for(; !itdag.isDone(); itdag.next())
		{
			MFnDagNode pf(itdag.currentItem());
			std::string curname = pf.name().asChar();
			//SHelper::stripAll(curname);
			if(curname == name) 
			{
				itdag.getPath(result);
				return 1;
			}
		}
	}
	
	return 0;
}

char ASearchHelper::dagByFullName(const char *name, MDagPath & res)
{
	std::string r;
	std::string str = name;
	typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
	boost::char_separator<char> sep("|/");
	tokenizer tokens(str, sep);
	MDagPath parent;
	for (tokenizer::iterator tok_iter = tokens.begin();
		tok_iter != tokens.end(); ++tok_iter)
	{
		r = *tok_iter;
		if(!dagDirectChildByName( parent, res, r))
			return 0;
		
		parent = res;
	}
	return 1;
}

void ASearchHelper::lastMesh(MObject & result)
{
	MStatus stat;
	MItDependencyGraph itdep(result, MFn::kMesh, MItDependencyGraph::kDownstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next()) {
		MObject ocur = itdep.currentItem();
		MFnDagNode fdep(ocur);
		
		if(!fdep.isIntermediateObject()) {
			result = ocur;
		}
	}
}

char ASearchHelper::sameParent(MObject &one, MObject &another)
{
    std::string nameOne(MFnDagNode(one).fullPathName().asChar());
    std::string nameTwo(MFnDagNode(another).fullPathName().asChar());
    nameOne = SHelper::getParentName(nameOne);
    nameTwo = SHelper::getParentName(nameTwo);
    return nameOne == nameTwo;
}

bool ASearchHelper::FirstTypedObj(const MObject &root, MObject & dst, MFn::Type typ)
{
    MStatus stat;
	MItDag iter;
	iter.reset(root, MItDag::kDepthFirst, typ);
	for(; !iter.isDone(); iter.next()) {								
		MDagPath apath;		
		iter.getPath( apath );
		MFnDagNode fdag(apath);
		if(!fdag.isIntermediateObject()) {
			dst = apath.node();
			return true;
		}
	}
    return false;
}

void ASearchHelper::LsAllTypedPaths(MDagPathArray & dst, const MDagPath & root, MFn::Type typ)
{
	MStatus stat;
	const unsigned nc = root.childCount();
	std::map<std::string, MDagPath > sorted;
// ls transform children ordered by name
	unsigned i;
	for(i=0; i<nc; i++) {
		MObject	child = root.child(i);
		if(child.hasFn(MFn::kTransform)) {
			MDagPath pchild;
			MDagPath::getAPathTo(child, pchild);
			sorted[MFnDagNode(child).name().asChar()] = pchild;
		}
	}
	
// handle transform children first
	std::map<std::string, MDagPath >::const_iterator it = sorted.begin();
	for(; it !=sorted.end(); ++it)
		LsAllTypedPaths(dst, it->second, typ);
	
// then typed children	
	if(root.node().hasFn(typ) && !MFnDagNode(root).isIntermediateObject()) 
		dst.append(root);
	
	sorted.clear();
	if(typ == MFn::kTransform) return;
	
	for(i=0; i<nc; i++) {
		MObject	child = root.child(i);
		if(child.hasFn(typ) && !MFnDagNode(child).isIntermediateObject()) {
			MDagPath pchild;
			MDagPath::getAPathTo(child, pchild);
			sorted[MFnDagNode(child).name().asChar()] = pchild;
		}
	}
	
	it = sorted.begin();
	for(; it !=sorted.end(); ++it)
		dst.append(it->second);
}

bool ASearchHelper::FirstObjByAttrValInArray(MObjectArray &objarr, MString &attrname, MString &attrval, MObject &res)
{
	MString meshname;
	for(unsigned i = 0; i <objarr.length(); i++) {
		if(getStringAttributeByName(objarr[i], attrname.asChar(), meshname)) {
			if(meshname == attrval) {
				res = objarr[i];
				return true;
			}
		}
		else {
			MPlug plg;
			if(FristNamedPlugInHistory(objarr[i], MFn::kMesh, attrname, plg)) {
				meshname = plg.asString();
				if(meshname == attrval) {
					res = objarr[i];
					return true;
				}
			}
		}
	}
	return false;
}

bool ASearchHelper::FristNamedPlugInHistory(MObject &root, MFn::Type type, MString &name1, MPlug &plug1)
{
	MStatus stat;
	MItDependencyGraph itdep(root, type, MItDependencyGraph::kUpstream, MItDependencyGraph::kDepthFirst, MItDependencyGraph::kNodeLevel, &stat );
	for(; !itdep.isDone(); itdep.next())
	{
	        MObject ocur = itdep.currentItem();
	        MFnDependencyNode fdep(ocur);
	        plug1 = fdep.findPlug( name1, false, &stat);
	        if(stat)
	                return true;
	}
	return false;
}

bool ASearchHelper::FirstDepNodeByName(MObject& node, const MString & name, MFn::Type type)
{
	MItDependencyNodes itdep(type);
	for(; !itdep.isDone(); itdep.next())
	{
		MObject ocur = itdep.thisNode();
		std::string curname(MFnDependencyNode(ocur).name().asChar());
		curname = SHelper::removeNamespace(curname);
		if(name == MString(curname.c_str()) )
		{
			node = ocur;
			return true;
		}
	}
	return false;
}

void ASearchHelper::TransformsBetween(MDagPathArray & dst,
								const MDagPath & longer, const MDagPath & shorter)
{
	MDagPath cur = longer;
	MStatus stat = cur.pop();
	while(stat) {
		if(cur == shorter) return;
		if(!cur.node().hasFn(MFn::kTransform)) return;
		dst.append(cur);
		stat = cur.pop();
	}
}

void ASearchHelper::TransformsToWorld(std::map<std::string, MDagPath> & dst,
								const MDagPath & longer)
{
	MDagPath cur = longer;
	MStatus stat = cur.pop();
	while(stat) {
		if(!cur.node().hasFn(MFn::kTransform)) return;
		dst[cur.fullPathName().asChar()] = cur;
		stat = cur.pop();
	}
}

void ASearchHelper::LsAllTransformsTo(std::map<std::string, MDagPath> & dst, const MDagPathArray & tails)
{
	const unsigned n = tails.length();
	unsigned i = 0;
	for(;i<n;i++)
		TransformsToWorld(dst, tails[i]);
}

void ASearchHelper::LsAll(std::map<std::string, MDagPath> & dst, const MDagPathArray & tails)
{
	const unsigned n = tails.length();
	unsigned i = 0;
	for(;i<n;i++)
		dst[tails[i].fullPathName().asChar()] = tails[i];
}
//~: