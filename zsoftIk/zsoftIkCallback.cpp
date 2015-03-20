/*
 *  zsoftIkCallback.cpp
 *  softIk
 *
 *  Created by jian zhang on 3/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include "zsoftIkCallback.h"
#include <maya/MGlobal.h>
#include <maya/MSelectionList.h>
//////////////////////////////////////////////////////////////////
//
// IK 2 Bone Solver Callbacks
//
//////////////////////////////////////////////////////////////////

MCallbackId addIK2BsolverCallbacks::afterNewId;
MCallbackId addIK2BsolverCallbacks::afterOpenId;                                                                                                                        

void *addIK2BsolverCallbacks::creator()
{
        return new addIK2BsolverCallbacks;
}

void createIK2BsolverAfterNew(void *clientData)
//
// This method creates the ik2Bsolver after a File->New.
//
{
        MSelectionList selList;
        MGlobal::getActiveSelectionList( selList );
        MGlobal::executeCommand("createNode -n zik2Bsolver zik2Bsolver");
        MGlobal::setActiveSelectionList( selList );
}

void createIK2BsolverAfterOpen(void *clientData)
//
// This method creates the ik2Bsolver after a File->Open
// if the ik2Bsolver does not exist in the loaded file.
//
{
        MSelectionList selList;
        MGlobal::getSelectionListByName("zik2Bsolver", selList);
        if (selList.length() == 0) {
                MGlobal::getActiveSelectionList( selList );
                MGlobal::executeCommand("createNode -n zik2Bsolver zik2Bsolver");
                MGlobal::setActiveSelectionList( selList );
        }
}

MStatus addIK2BsolverCallbacks::doIt(const MArgList &args)
//
// This method adds the File->New and File->Open callbacks
// used to recreate the ik2Bsolver.
//
{
    // Get the callback IDs so we can deregister them 
        // when the plug-in is unloaded.
        afterNewId = MSceneMessage::addCallback(MSceneMessage::kAfterNew, 
                                                           createIK2BsolverAfterNew);
        afterOpenId = MSceneMessage::addCallback(MSceneMessage::kAfterOpen, 
                                                           createIK2BsolverAfterOpen);
        return MS::kSuccess;
}


