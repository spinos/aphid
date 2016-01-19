/*
 *  StickyManip.h
 *  manuka
 *
 *  Created by jian zhang on 1/18/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */

#include <maya/MPxLocatorNode.h> 
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
	MDagPath fNodePath;

public:
    static MTypeId id;
};

class CircleCurve;

class StickyLocator : public MPxLocatorNode
{
	CircleCurve * m_circle;
	MPoint m_origin;
	
public:
	StickyLocator();
	virtual ~StickyLocator(); 

    virtual MStatus   		compute(const MPlug& plug, MDataBlock &data);

	virtual void            draw(M3dView &view, const MDagPath &path, 
								 M3dView::DisplayStyle style,
								 M3dView::DisplayStatus status);

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 

	static  void *          creator();
	static  MStatus         initialize();

	static MObject         size;
	static MObject aMoveVX;
	static MObject aMoveVY;
	static MObject aMoveVZ;
	static MObject aMoveV;
	static MObject ainmesh;
	static MObject avertexId;
	static MObject aoutMeanX;
	static MObject aoutMeanY;
	static MObject aoutMeanZ;
	static MObject aoutMean;
	static  	MObject 	ainrefi;
	static  	MObject 	ainrefd;
	
public: 
	static	MTypeId		id;
	
private:
	void drawCircle() const;
};
