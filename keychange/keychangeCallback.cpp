/*
 *  zsoftIkCallback.cpp
 *  softIk
 *
 *  Created by jian zhang on 3/20/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *  
 *  how to use:
 *   select animated object
 *   keychangeCallbacks
 *  log:
 *   //  tracking keyframe change of pSphere1_translateX // 
 *   //  tracking keyframe change of pSphere1_translateY // 
 *   //  tracking keyframe change of pSphere1_translateZ //
 *   // pSphere1_translateZ is edited! // 
 *   //  key 1 is edited! // 
 *   //  key 1 is edited! // 
 *   //  key 1 is edited! // 
 */

#include "keychangeCallback.h"
#include <maya/MGlobal.h>
#include <maya/MSelectionList.h>
#include <maya/MFnKeyframeDelta.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MSelectionList.h>
#include <maya/MAnimUtil.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>
#include <maya/MObjectArray.h>

MCallbackId keychagneCallbacks::keyEditedId;                                                                                                                      

void *keychagneCallbacks::creator()
{
        return new keychagneCallbacks;
}

void printKeyDelta(MObject & node, MObjectArray &objects, void *clientData)
{
    MGlobal::displayInfo(MFnDependencyNode(node).name() + " is edited!");
    unsigned i;
    for(i=0; i < objects.length(); i++) {
        MFnKeyframeDelta fdelta(objects[i]);
        MGlobal::displayInfo(MString(" key ") + fdelta.keyIndex() + " is edited!");
    }
}

MStatus keychagneCallbacks::doIt(const MArgList &args)
{
    MSelectionList sell;
    MGlobal::getActiveSelectionList(sell);
    MPlugArray animatedPlugs;
    MAnimUtil::findAnimatedPlugs (sell, animatedPlugs);
    
    MObjectArray animation;
    unsigned i;
    for(i=0; i< animatedPlugs.length(); i++ ) {
         MPlug plug = animatedPlugs[i];
         
        // Find the animation curve(s) that animate this plug
        //
        MAnimUtil::findAnimation (plug, animation);
    }
    
    for(i=0; i <animation.length(); i++) {
    
        MObject animNode = animation[i];
        MStatus status;
        keyEditedId = MAnimMessage::addNodeAnimKeyframeEditedCallback(animNode, printKeyDelta,  NULL, &status);

        if(status) MGlobal::displayInfo(MString(" tracking keyframe change of ") + MFnDependencyNode(animNode).name());
    }
    return MS::kSuccess;
}


