/*
 *  StickyManip.h
 *  manuka
 *
 *  Created by jian zhang on 1/18/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once

#include <maya/MPxManipContainer.h> 
#include <maya/MManipData.h>

class StickyLocatorManip : public MPxManipContainer
{
public:
    StickyLocatorManip();
    virtual ~StickyLocatorManip();
    
    static void * creator();
    static MStatus initialize();
    virtual MStatus createChildren();
    virtual MStatus connectToDependNode(const MObject & node);

    virtual void draw(M3dView & view, 
					  const MDagPath & path, 
					  M3dView::DisplayStyle style,
					  M3dView::DisplayStatus status);
	MManipData startPointCallback(unsigned index) const;
	MVector nodeTranslation() const;

    MDagPath fDistanceManip;
	MDagPath fDirectionManip;
	MDagPath fDropoffManip;
	MDagPath fNodePath;

public:
    static MTypeId id;
};

