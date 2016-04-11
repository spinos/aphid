/*
 *  ExampVox.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/5/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#include <DrawBox.h>
#include <BoundingBox.h>
#include <Geometry.h>
#include <vector>
#include <VectorArray.h>
#include <ConvexShape.h>
#include <GridClustering.h>
#include <AOrientedBox.h>
#include <boost/scoped_ptr.hpp>

namespace aphid {

class ExampVox : public DrawBox {

	BoundingBox m_geomBox;
	Vector3F m_geomCenter;
	Vector3F * m_boxNormalBuf;
	Vector3F * m_boxPositionBuf;
	boost::scoped_ptr<Vector3F> m_dopNormalBuf;
	boost::scoped_ptr<Vector3F> m_dopPositionBuf;
	float * m_boxCenterSizeF4;
	float m_diffuseMaterialColV[3];
	float m_geomScale[3];
/// radius of bbox
	float m_geomExtent;
/// radius exclusion
	float m_geomSize;
/// scaling radius of exclusion
	float m_sizeMult;
	unsigned m_numBoxes;
	unsigned m_boxBufLength;
	int m_dopBufLength;
	
public:
	ExampVox();
	virtual ~ExampVox();
	
	virtual void voxelize(const std::vector<Geometry *> & geoms);
	virtual void voxelize1(sdb::VectorArray<cvx::Triangle> * tri,
							const BoundingBox & bbox);
	
/// set b4 geom box
	void setGeomSizeMult(const float & x);
	void setGeomBox(const float & a, 
					const float & b,
					const float & c,
					const float & d,
					const float & e,
					const float & f);
	
	const float & geomExtent() const;
	const float & geomSize() const;
	const BoundingBox & geomBox() const;
	const float * geomCenterV() const;
	const Vector3F & geomCenter() const;
	const float * geomScale() const;
	const float * diffuseMaterialColor() const;
	const float * boxCenterSizeF4() const;
	const unsigned & numBoxes() const;
	const float * boxNormalBuf() const;
	const float * boxPositionBuf() const;
	const unsigned & boxBufLength() const;
	const int & dopBufLength() const;
	
protected:
	void drawDop();
	void drawGrid();
	void drawWireGrid();
	float * diffuseMaterialColV();
	float * boxCenterSizeF4();
	bool setNumBoxes(unsigned n);
	void buildBoxDrawBuf();
	void buildDOPDrawBuf(const sdb::VectorArray<AOrientedBox> & dops);
	
private:
	void fillGrid(sdb::WorldGrid<GroupCell, unsigned > * grid,
				Geometry * geo);
};

}