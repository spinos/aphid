#pragma once
#include <AllMath.h>
#include <BoundingRectangle.h>
class MlRachis;
class CollisionRegion;
class MlFeather {
public:
    MlFeather();
    virtual ~MlFeather();
    void createNumSegment(short x);

	short numSegment() const;
	unsigned numVaneVertices() const;
	unsigned numWorldP() const;
    float * quilly();
    float * getQuilly() const;
	Vector2F baseUV() const;
	
	Vector2F * vane();
    Vector2F * vaneAt(short seg, short side);
    Vector2F * getVaneAt(short seg, short side) const;
	float getLength() const;
	float getWidth(short seg) const;
	BoundingRectangle getBoundingRectangle() const;
	
	void computeWorldP(const Vector3F & oriPos, const Matrix33F & oriRot, const float& pitch, const float & scale);
	Vector3F * segmentOriginWP(short seg);
	Vector3F * segmentVaneWP(short seg, short side, short idx);
	
	Vector3F getSegmentOriginWP(short seg) const;
	Vector3F getSegmentVaneWP(short seg, short side, short idx) const;
	
	void setCollision(CollisionRegion * skin);
	
	void setFeatherId(short x);
	short featherId() const;
	void setBaseUV(const Vector2F & d);
	void translateUV(const Vector2F & d);
	
	void computeLength();
	void computeBounding();
	
	float* selectVertexInUV(const Vector2F & p, bool & yOnly, Vector2F & wp);
	
	void verbose();
private:
	void defaultCreate(int ns = 5);
	void computeVaneWP(const Vector3F & origin, const Matrix33F& space, short seg, short side, float scale);
private:
	BoundingRectangle m_brect;
	Vector2F m_uv;
	CollisionRegion * m_skin;
	MlRachis * m_rachis;
    float *m_quilly;
    Vector2F * m_vaneVertices;
	Vector3F * m_worldP;
	float m_length;
    short m_numSeg, m_id;
};
