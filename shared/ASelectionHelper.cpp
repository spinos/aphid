/*
 *  selectionUtil.cpp
 *  opium
 *
 *  Created by jian zhang on 6/3/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#include "ASelectionHelper.h"
#include <foundation/SHelper.h>

namespace aphid {
    
void SelectionHelper::setBehead(const MString &name)
{
	SHelper::getHierarchy(name.asChar(), _behead_list);
}

void SelectionHelper::excludeHeads(MDagPathArray &arr)
{
	if(_behead_list.size()<1)
		return;
		
	for(unsigned i=0; i<arr.length(); i++)
	{
		std::string pull = arr[i].fullPathName().asChar();
		if(pull.find(_behead_list[0]) == 0)
		{
			for(unsigned j=0; j<_behead_list.size(); j++)
			{
				if(pull == _behead_list[j])
				{
					arr.remove(i);
					j = 99;
					i--;
				}
			}
		}
	}
}

void SelectionHelper::getEveryObj(MObjectArray &active_list, MObject &root)
{
	MItDag itdag(MItDag::kDepthFirst);
	
	if(root != MObject::kNullObj)
		itdag.reset(root);
				
	for(; !itdag.isDone(); itdag.next())
	{
		active_list.append(itdag.currentItem());
	}
}

void SelectionHelper::getEverything(MDagPathArray &active_list, MObject &root)
{
	MItDag itdag(MItDag::kDepthFirst);
	
	if(root != MObject::kNullObj)
		itdag.reset(root);
				
	for(; !itdag.isDone(); itdag.next())
	{
		MDagPath apath;
		itdag.getPath(apath);
		active_list.append(apath);
	}
	reportNumActive(active_list);
	excludeHeads(active_list);
}

char SelectionHelper::getAll(MDagPathArray &active_list)
{
    MGlobal::displayInfo("get all ancestor and descendant of selected and selected");  
    _nameMap.clear();
	MSelectionList selList;
	MGlobal::getActiveSelectionList ( selList );
	if ( isEmptySelection(selList) ) 
	    return 0;
	
	MItSelectionList iter( selList );
	for ( ; !iter.isDone(); iter.next() ) {								
		MDagPath apath;		
		iter.getDagPath( apath );
		
		if(!AHelper::containsGeom(apath)) {				
            MGlobal::displayInfo(apath.fullPathName() + " has no geo child");
            continue;
        }
        
        getParents(apath, active_list);
        active_list.append(apath);
		getChildren(apath, active_list);
	}
	
	excludeHeads(active_list);	
	reportNumActive(active_list);
	
	return 1;
}

char SelectionHelper::getUpper(MDagPathArray &active_list)
{
    MGlobal::displayInfo("get all ancestor of selected and selected");
    _nameMap.clear();
	MSelectionList selList;
	MGlobal::getActiveSelectionList ( selList );
	if ( isEmptySelection(selList) ) 
	    return 0;
	
	MItSelectionList iter( selList );
	
	for ( ; !iter.isDone(); iter.next() ) {								
		MDagPath apath;		
		iter.getDagPath( apath );
		getParents(apath, active_list);
        active_list.append(apath);
	}
	
	excludeHeads(active_list);
	reportNumActive(active_list);
	return 1;
}

char SelectionHelper::getDown(MDagPathArray &active_list)
{
    MGlobal::displayInfo("get all descendant of selected and selected");
	
	MSelectionList selList;
	MGlobal::getActiveSelectionList ( selList );
	if ( isEmptySelection(selList) ) 
	    return 0;
	
	MItSelectionList iter( selList );
	
	for ( ; !iter.isDone(); iter.next() ) 
	{								
		MDagPath apath;		
		iter.getDagPath( apath );
		active_list.append(apath);
		getChildren(apath, active_list);	
	}
	
	excludeHeads(active_list);
	reportNumActive(active_list);
	return 1;
}

void SelectionHelper::getDepnode(MObjectArray &arr)
{
	MSelectionList selList;
	MGlobal::getActiveSelectionList ( selList );
	if ( isEmptySelection(selList) ) 
	    return;
	MItSelectionList iter( selList );

	for ( ; !iter.isDone(); iter.next() ) 
	{								
		MObject anode;		
		iter.getDependNode( anode );

		arr.append(anode);	
	}
}

char SelectionHelper::getSingleMesh(MDagPath &mesh, MObject &component)
{
	MSelectionList selList;
	MGlobal::getActiveSelectionList ( selList );
	if ( isEmptySelection(selList) ) 
	    return 0;
	
	MItSelectionList iter( selList );

	for ( ; !iter.isDone(); iter.next() ) 
	{								
		iter.getDagPath( mesh, component );
		mesh.extendToShape();
		if(mesh.hasFn(MFn::kMesh))
			return 1;
	}
	return 0;
}

char SelectionHelper::meshFaceById(MObject & mesh, const int count, const int *idx)
{
	MStatus stat;
	MFnDagNode fdag(mesh, &stat);
	if(!stat)
		MGlobal::displayInfo("no dag node");
		
	MDagPath pmesh;
	fdag.getPath(pmesh);
		
	MItMeshPolygon faceIt(mesh, &stat);
	if(!stat)
		MGlobal::displayInfo("no iter");
		
	MSelectionList sell;
	int preidx;
	for(int i = 0; i < count; i++)
	{
		faceIt.setIndex(idx[i], preidx);
		MObject comp = faceIt.currentItem();
		sell.add(pmesh, comp);
	}
	MGlobal::setActiveSelectionList(sell);
	return 1;
}

void SelectionHelper::getParents(MDagPath & child, MDagPathArray &active_list)
{
    std::string aname(child.fullPathName().asChar());
    std::vector<std::string> parents;
    SHelper::listAllNames( aname, parents );
    
    for(int par_iter = 0; par_iter < parents.size() - 1; par_iter++) {				
        if(_nameMap.count(parents[par_iter]) < 1) {
            MObject oparent = AHelper::getTypedObjByFullName(MFn::kTransform, parents[par_iter].c_str());
            MFnDagNode fparent(oparent);
			MDagPath parentPath;
            fparent.getPath(parentPath);
            active_list.append(parentPath);
            _nameMap[parents[par_iter]] = parents[par_iter];
        }
    }
}

void SelectionHelper::getChildren(MDagPath & root, MDagPathArray &active_list)
{
    MItDag itdag(MItDag::kBreadthFirst);
    itdag.reset (root);
    
    char first = 1;    
    for(; !itdag.isDone(); itdag.next()) {
        if(first) {
            first = 0;
            continue;
        }
        MDagPath childPath;
        itdag.getPath(childPath);
        active_list.append(childPath);
    }
}

char SelectionHelper::isEmptySelection(MSelectionList & sels)
{
    MGlobal::displayInfo(MString("num selection ") + sels.length());
    return ( sels.length() == 0 );
}

void SelectionHelper::reportNumActive(MDagPathArray &arr)
{
    MGlobal::displayInfo(MString("num active entities ") + arr.length());
}

char SelectionHelper::getSelected(MDagPathArray &active_list)
{
    MSelectionList selList;
	MGlobal::getActiveSelectionList ( selList );
	if ( isEmptySelection(selList) ) 
	    return 0;
	
	MItSelectionList iter( selList );
	for ( ; !iter.isDone(); iter.next() ) {								
		MDagPath apath;		
		iter.getDagPath( apath );

        active_list.append(apath);
	}
	
	excludeHeads(active_list);	
	reportNumActive(active_list);
	
	return 1;
}

}
//:~
