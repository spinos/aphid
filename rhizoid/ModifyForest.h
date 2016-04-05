/*
 *  ModifyForest.h
 *  proxyPaint
 *
 *  Created by jian zhang on 2/1/16.
 *  Copyright 2016 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include "Forest.h"
#include <PseudoNoise.h>

namespace aphid {

class TriangleRaster;
class BarycentricCoordinate;

class ModifyForest : public Forest {
	
	TriangleRaster * m_raster;
	BarycentricCoordinate * m_bary;
	PseudoNoise m_pnoise;
	int m_seed;
	float m_noiseWeight;
    
public:
    struct GrowOption {
		Vector3F m_upDirection;
		Vector3F m_centerPoint;
		int m_plantId;
		float m_minScale, m_maxScale;
		float m_minMarginSize, m_maxMarginSize;
		float m_rotateNoise;
		float m_strength;
		float m_radius;
		bool m_alongNormal;
		bool m_multiGrow;
	};
    
	ModifyForest();
	virtual ~ModifyForest();
	
    void setNoiseWeight(float x);
    void erectActive();
	
protected:
	bool growOnGround(GrowOption & option);
	
	bool growAt(const Ray & ray, GrowOption & option);
	bool growAt(const Matrix44F & trans, GrowOption & option);
	void replaceAt(const Ray & ray, GrowOption & option);
	void clearAt(const Ray & ray, GrowOption & option);
	void scaleAt(const Ray & ray, float magnitude);
    void rotateAt(const Ray & ray, float magnitude, int axis);
	void movePlant(const Ray & ray,
					const Vector3F & displaceNear, const Vector3F & displaceFar,
					const float & clipNear, const float & clipFar);
	void moveWithGround();
	void scaleBrushAt(const Ray & ray, float magnitude);
    
private:
	void growOnFaces(Geometry * geo, sdb::Sequence<unsigned> * components, 
					int geoId,
					GrowOption & option);
					
	void growOnFace(const int & geoId, const int & triId,
					GrowOption & option);
	
	void growOnTriangle(TriangleRaster * tri, 
					BarycentricCoordinate * bar,
					GroundBind & bind,
					GrowOption & option);
	void randomSpaceAt(const Vector3F & pos, 
							const GrowOption & option,
							Matrix44F & space, float & scale);
	void movePlantsWithGround(sdb::Array<int, Plant> * arr);
	bool calculateSelecedWeight(const Ray & ray);
    float getNoise() const;
    float getNoise2(const float & a, const float & b) const;
    
};

}