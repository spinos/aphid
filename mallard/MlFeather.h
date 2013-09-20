#pragma once
#include <AllMath.h>
class MlRachis;
class MlFeather {
public:
    MlFeather();
    virtual ~MlFeather();
    void createNumSegment(short x);
	void computeLength();
    
	short numSegment() const;
    float * quilly();
    float * getQuilly() const;
    Vector2F * vaneAt(short seg, short side);
    Vector2F * getVaneAt(short seg, short side) const;
	float getLength() const;
	float getWidth(short seg) const;
	
	void computeWorldP(const Vector3F & oriPos, const Matrix33F & oriRot, const float& pitch, const float & scale);
	Vector3F * segmentOriginWP(short seg);
	Vector3F * segmentVaneWP(short seg, short side, short idx);
	
	Vector3F getSegmentOriginWP(short seg) const;
	Vector3F getSegmentVaneWP(short seg, short side, short idx) const;
private:
	void computeVaneWP(const Vector3F & origin, const Matrix33F& space, short seg, short side, float scale);
private:
	MlRachis * m_rachis;
    float *m_quilly;
    Vector2F * m_vaneVertices;
	Vector3F * m_worldP;
	float m_length;
    short m_numSeg;
};
