/*
 *  AOrientedBox.h
 *  aphid
 *
 *  Created by jian zhang on 8/24/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef APH_ORIENTED_BOX_H
#define APH_ORIENTED_BOX_H

#include <math/Matrix33F.h>
#include <math/BoundingBox.h>

namespace aphid {

class AOrientedBox {
public:
	AOrientedBox();
	virtual ~AOrientedBox();
	
	void setCenter(const Vector3F & p);
	void setOrientation(const Matrix33F & m,
					const Matrix33F::RotateOrder & rod = Matrix33F::XYZ);
	void setExtent(const Vector3F & p,
					const Matrix33F::RotateOrder & rod = Matrix33F::XYZ);
	void set8DOPExtent(const float & x0, const float & x1,
						const float & y0, const float & y1);
	void calculateCenterExtents(const float * p,
					const int & np,
					const BoundingBox * box);
/// longest axis as x
/// 2nd longest axis as y
	void caluclateOrientation(const BoundingBox * box);
/// sx[4] dop xy move around .707 value [-1, 1] 
	void calculateCenterExtents(const BoundingBox * box,
					const float * sx);
	
	const Vector3F & center() const;
	const Matrix33F & orientation() const;
	const Vector3F & extent() const;
	const float * dopExtent() const;
	Vector3F majorPoint(bool low) const;
	Vector3F majorVector(bool low) const;
    Vector3F minorPoint(bool low) const;
	Vector3F minorVector(bool low) const;
	void getBoxVertices(Vector3F * dst) const;
	
	const BoundingBox calculateBBox() const;
	void limitMinThickness(const float & x);

	Vector3F get8DOPFaceX() const;
	Vector3F get8DOPFaceY() const;
	
/// transform points to local space
/// scale to [-1, 1]
	template<typename T>
	void projectToLocalUnit(T & pnts, const int & npnts) const
	{
		Matrix33F localspace(m_orientation);
		localspace.inverse();
		std::cout<<" inv rot "<<localspace;
		Vector3F scaling = m_extent.inversed();
		std::cout<<" inv scale "<<scaling;
		
		for(int i=0;i<npnts;++i) {
			Vector3F plocal = pnts[i] - m_center;
			plocal = localspace.transform(plocal);
			plocal *= scaling;
			pnts[i] = plocal;
		}
	}
	
	friend std::ostream& operator<<(std::ostream &output, const AOrientedBox & p);
		
protected:

private:
	float remap(float x);

private:
	Matrix33F m_orientation;
	Vector3F m_center;
	Vector3F m_extent;
/// in xy
	float m_8DOPExtent[4];
};

}
#endif