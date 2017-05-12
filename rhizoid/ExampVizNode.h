/*
 *  ExampVizNode.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include "ExampVox.h"
#include <ogl/DrawCircle.h>
#include <maya/MPxLocatorNode.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/M3dView.h>
#include <maya/MVectorArray.h>

namespace aphid {

class BoundingBox;

template<typename T>
class DenseMatrix;

namespace sdb {
template<typename T>
class ValGrid;
}

}

class ExampViz : public MPxLocatorNode, public aphid::ExampVox, public aphid::DrawCircle
{
	float m_transBuf[16];
	float m_preDiffCol[3];
	float m_preDspSize[3];
	
public:
	ExampViz();
	virtual ~ExampViz(); 

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

	static  MObject         abboxminv;
	static  MObject         abboxmaxv;
	static  MObject         adoplen;
	static  MObject         adopPBuf;
	static  MObject         adopNBuf;
	static  MObject         adopCBuf;
	static MObject adrawColorR;
	static MObject adrawColorG;
	static MObject adrawColorB;
	static MObject adrawColor;
	static MObject adrawDopSizeX;
	static MObject adrawDopSizeY;
	static MObject adrawDopSizeZ;
	static MObject adrawDopSize;
	static MObject aradiusMult;
	static MObject aininstspace;
	static MObject avoxactive;
	static MObject avoxvisible;
	static MObject avoxpriority;
	static MObject adrawVoxTag;
	static MObject outValue;
	static	MTypeId		id;
						
	virtual void voxelize3(aphid::sdb::VectorArray<aphid::cvx::Triangle> * tri,
							const aphid::BoundingBox & bbox);
	
	void voxelize3(const aphid::DenseMatrix<float> & pnts,
						const MIntArray & triangleVertices,
						const aphid::BoundingBox & bbox);
/// point sample based before grid faces 
	void voxelize4(aphid::sdb::VectorArray<aphid::cvx::Triangle> * tri,
						const aphid::BoundingBox & bbox);
	typedef aphid::sdb::ValGrid<aphid::PosNmlCol > VGDTyp;					
private:
	void updateGeomBox(MObject & node);
	bool loadTriangles(MDataBlock & data);
	bool loadTriangles(MObject & node);
	void fillDefaultCol(MVectorArray & cols,
					int n);
	void buildDrawBuf(int n,
				const MVectorArray & pnts,
				const MVectorArray & nmls,
				const MVectorArray & cols);
	void updateGridUniformColor(const float * col);
/// size and color if changed
	void updateDop();
	
	
};