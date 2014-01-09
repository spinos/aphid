#pragma once
#include <AllMath.h>
#include "DeformableFeather.h"
#include <BoundingBox.h>
class MlRachis;
class CollisionRegion;
class MlVane;
class MlFeather : public DeformableFeather {
public:
    MlFeather();
    virtual ~MlFeather();
    virtual void createNumSegment(short x);
	virtual void changeNumSegment(int d);
	virtual void computeLength();
	virtual void createVanes();

	void bend();
	void bendAt(unsigned faceIdx, float patchU, float patchV, const Vector3F & oriPos, const Matrix33F & oriRot, const float & scale);
	void curl(float val);
	
	void computeWorldP(const Vector3F & oriPos, const Matrix33F & oriRot, const float & scale);
	Vector3F * segmentVaneWP(short u, short v, short side);
	
	void setCollision(CollisionRegion * skin);
	
	void setFeatherId(short x);
	short featherId() const;
	
	void getBoundingBox(BoundingBox & box);
	float * angles() const;
	float bendDirection() const;
	void verbose();
	
	MlVane * vane(short side) const;
	
	void computeNoise();
	void samplePosition(Vector3F * dst);
	
	void setSeed(unsigned s);
	void setNumSeparate(unsigned n);
	void setSeparateStrength(float k);
	void setFuzzy(float f);
	
	unsigned seed() const;
	unsigned numSeparate() const;
	float fuzzy() const;
	float separateStrength() const;
	
	void testVane();
	void separateVane();
	
	void samplePosition(float lod);
	float scaledShaftLength() const;
	
	Vector3F * patchCenterP(short seg);
	Vector3F * patchWingP(short seg, short side);
protected:	
	virtual void simpleCreate(int ns = 5);
	
private:
	void computeVaneWP(const Vector3F & origin, const Matrix33F& space, short seg, float xscale);
	void samplePosition(unsigned nu, unsigned nv, int side, float lod);
private:
	CollisionRegion * m_skin;
	MlRachis * m_rachis;
	MlVane * m_vane;
	float m_scale;
	short m_id;
};
