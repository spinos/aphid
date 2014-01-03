#pragma once
#include "TexturedFeather.h"
#include <BoundingBox.h>
class MlRachis;
class CollisionRegion;
class MlVane;
class MlFeather : public TexturedFeather {
public:
    MlFeather();
    virtual ~MlFeather();
    virtual void createNumSegment(short x);
	virtual void changeNumSegment(int d);
	virtual void computeLength();
	
	void setupVane();
	
	void bend();
	void bendAt(unsigned faceIdx, float patchU, float patchV, const Vector3F & oriPos, const Matrix33F & oriRot, const float & scale);
	void curl(float val);
	
	void computeWorldP(const Vector3F & oriPos, const Matrix33F & oriRot, const float & scale);
	Vector3F * worldP();
	Vector3F * segmentOriginWP(short seg);
	Vector3F * segmentVaneWP(short seg, short side, short idx);
	Vector3F * segmentVaneWP1(unsigned seg, unsigned end, short side);
	
	Vector3F getSegmentOriginWP(short seg) const;
	Vector3F getSegmentVaneWP(short seg, short side, short idx) const;
	
	void setCollision(CollisionRegion * skin);
	
	void setFeatherId(short x);
	short featherId() const;
	
	void getBoundingBox(BoundingBox & box) const;
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
	
	unsigned numSeparate() const;
	float fuzzy() const;
	float separateStrength() const;
	
	void testVane();
	void separateVane();
	
	void samplePosition(float lod);
	float scaledShaftLength() const;
protected:	
	virtual void simpleCreate(int ns = 5);
	
private:
	void computeVaneWP(const Vector3F & origin, const Matrix33F& space, short seg, short side, float scale);
	void samplePosition(unsigned nu, unsigned nv, int side);
private:
	CollisionRegion * m_skin;
	MlRachis * m_rachis;
    Vector3F * m_worldP;
	MlVane * m_vane;
	unsigned m_numSeparate;
	float m_fuzzy, m_separateStrength, m_scale;
	short m_id;
};
