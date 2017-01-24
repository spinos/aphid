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

namespace aphid {

class ForestCell;
class PseudoNoise;
class BarycentricCoordinate;
class EbpGrid;

class ModifyForest : public Forest {
    
public:
    enum ManipulateMode {
		manNone = 0,
		manRotate = 1
	};
	
private:
	BarycentricCoordinate * m_bary;
	PseudoNoise * m_pnoise;
	int m_seed;
	float m_noiseWeight;
	ManipulateMode m_manipulateMode;
	
public:
    struct GrowOption {
		Vector3F m_upDirection;
		Vector3F m_centerPoint;
		Vector3F m_noiseOrigin;
		int m_plantId;
		float m_minScale, m_maxScale;
		float m_minMarginSize, m_maxMarginSize;
		float m_rotateNoise;
		float m_strength;
		float m_radius;
		float m_noiseFrequency;
		float m_noiseLacunarity;
		float m_noiseLevel;
		float m_noiseGain;
		int m_noiseOctave;
		bool m_alongNormal;
		bool m_multiGrow;
		bool m_stickToGround;
		float m_strokeMagnitude;
		
		GrowOption() {
			m_upDirection = Vector3F::YAxis;
			m_alongNormal = 0;
			m_minScale = 1.f;
			m_maxScale = 1.f;
			m_rotateNoise = 0.f;
			m_plantId = 0;
			m_multiGrow = 1;
			m_minMarginSize = .1f;
			m_maxMarginSize = .1f;
			m_strength = .67f;
			m_stickToGround = true;
			m_noiseFrequency = 1.f;
			m_noiseLacunarity = 1.5f;
			m_noiseOctave = 4;
			m_noiseLevel = 0.f;
			m_noiseGain = .5f;
			m_noiseOrigin.set(.4315f, .63987f, .6589f);
		}
		
		void setStrokeMagnitude(const float & x) 
		{
			m_strokeMagnitude = x;
			if(m_strokeMagnitude < -.5f)
				m_strokeMagnitude = -.5f;
			if(m_strokeMagnitude > .5f)
				m_strokeMagnitude = .5f;
		}
	};
    
	ModifyForest();
	virtual ~ModifyForest();
	
    void setNoiseWeight(float x);
    void rightUp(GrowOption & option);
	void scalePlant(GrowOption & option);
	void movePlant(GrowOption & option);
    void rotatePlant(GrowOption & option);
    void removeActivePlants();
    void removeTypedPlants(int x);
	void clearPlantOffset(GrowOption & option);
	void setManipulatMode(ManipulateMode x);
	ManipulateMode manipulateMode() const;
	
protected:
	bool growOnGround(GrowOption & option);
	
	bool growAt(const Ray & ray, GrowOption & option);
	bool growAt(const Matrix44F & trans, GrowOption & option);
	void replaceAt(const Ray & ray, GrowOption & option);
	void clearSelected();
	void clearAt(const Ray & ray, GrowOption & option);
	void scaleAt(const Ray & ray, float magnitude,
				bool isBundled);
	void rotateAt(const Ray & ray, float magnitude, int axis);
	void movePlant(const Ray & ray,
					const Vector3F & displaceNear, const Vector3F & displaceFar,
					const float & clipNear, const float & clipFar);
	void rotatePlant(const Ray & ray,
					const Vector3F & displaceNear, const Vector3F & displaceFar,
					const float & clipNear, const float & clipFar);
/// use delta rotation
	void rotatePlant();

	void moveWithGround();
	void scaleBrushAt(const Ray & ray, float magnitude);
	void raiseOffsetAt(const Ray & ray, GrowOption & option);
	void calculateSelectedWeight();
	virtual void getDeltaRotation(Matrix33F & mat,
					const float & weight = 1.f) const;
				
private:
	void clearPlant(Plant * pl, const sdb::Coord2 & k);
	void randomSpaceAt(const Vector3F & pos, 
							const GrowOption & option,
							Matrix44F & space, float & scale);
	void movePlantsWithGround(ForestCell * arr);
	bool calculateSelecedWeight(const Ray & ray);
    float getNoise() const;
    float getNoise2(const float & a, const float & b) const;
    bool sampleGround(EbpGrid * sampler, GrowOption & option);
/// when bundleBegin > -1
/// does not collide plant id >= bundleBegin
	bool growSingle(GrowOption & option,
				GroundBind & bind,
				const int & iExample,
				const Matrix44F & tm,
				CollisionContext * collctx);
	void growBundle(GrowOption & option,
				GroundBind & bind,
				const ExampVox * bundle,
				const int & iExample,
				const Matrix44F & tm,
				CollisionContext * collctx);
				
};

}