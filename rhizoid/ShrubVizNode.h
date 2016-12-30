#ifndef SHRUB_VIZ_NODE_H
#define SHRUB_VIZ_NODE_H

/*
 *  ShrubVizNode.h
 *  proxyPaint
 *
 *  Created by jian zhang on 3/1/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
 
#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>
#include <maya/MDagPath.h>
#include <maya/MSceneMessage.h>
#include <maya/MIntArray.h>
#include <maya/MVectorArray.h>
#include <ogl/DrawBox.h>
#include <ogl/DrawInstance.h>
#include <vector>

namespace aphid {

class ExampVox;

template<typename T>
class DenseMatrix;

class BoundingBox;

class ShrubVizNode : public MPxLocatorNode, public DrawBox, public DrawInstance
{
	struct InstanceD {
		float _trans[16];
		int _exampleId;
		int _instanceId;
	};
	
	std::vector<InstanceD > m_instances;
	std::vector<ExampVox * > m_examples;
	
public:
	ShrubVizNode();
	virtual ~ShrubVizNode(); 

    virtual MStatus   		compute( const MPlug& plug, MDataBlock& data );

	virtual void            draw( M3dView & view, const MDagPath & path, 
								  M3dView::DisplayStyle style,
								  M3dView::DisplayStatus status );

	virtual bool            isBounded() const;
	virtual MBoundingBox    boundingBox() const; 
	virtual MStatus connectionMade ( const MPlug & plug, const MPlug & otherPlug, bool asSrc );
	virtual MStatus connectionBroken ( const MPlug & plug, const MPlug & otherPlug, bool asSrc );
	
	static  void *          creator();
	static  MStatus         initialize();

	static MTypeId id;
	static MObject ashrubbox;
	static MObject ainsttrans;
	static MObject ainstexamp;
	static MObject ainexamp;
    static MObject outValue;
	
	const MMatrix & worldSpace() const;
	
	void setBBox(const BoundingBox & bbox);
/// instance as transform 4-by-4 and example_id
	void addInstance(const DenseMatrix<float> & trans,
					const int & exampleId);
	
protected:
	void getBBox(BoundingBox & bbox) const;
	int numInstances() const;
	int numExamples() const;
	void drawWiredBoundInstances() const;
	void drawSolidInstances() const;
	void drawWiredInstances() const;
	   
private:
	void attachSceneCallbacks();
	void detachSceneCallbacks();
	static void releaseCallback(void* clientData);

	MCallbackId fBeforeSaveCB;
	void saveInternal();
	bool loadInstances(const MVectorArray & instvecs,
						const MIntArray & instexmps);
	bool loadInternal();
	bool loadInternal(MDataBlock& block);
	void addExample(const MPlug & plug);
	
};

}
#endif        //  #ifndef ShrubVizNode_H
