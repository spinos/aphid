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
	void updateVane();
	
	void samplePosition(unsigned gridU, unsigned gridV, Vector3F * dst);
	void setSeed(unsigned s);
	void setNumSeparate(unsigned n);
	void setSeparateStrength(float k);
	void setFuzzy(float f);
	void setGrid(unsigned gridShaft, unsigned gridBarb);
protected:	
	virtual void simpleCreate(int ns = 5);
	
private:
	void computeVaneWP(const Vector3F & origin, const Matrix33F& space, short seg, short side, float scale);

private:
	CollisionRegion * m_skin;
	MlRachis * m_rachis;
    Vector3F * m_worldP;
	MlVane * m_vane;
	short m_id;
};
