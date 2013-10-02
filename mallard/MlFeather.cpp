#include "MlFeather.h"
#include "MlRachis.h"
#include <CollisionRegion.h>
MlFeather::MlFeather() : m_quilly(0), m_vaneVertices(0), m_worldP(0) 
{
	m_rachis = new MlRachis;
	m_uv.set(4.f, 4.f);
	defaultCreate();
}

MlFeather::~MlFeather() 
{
    if(m_quilly) delete[] m_quilly;
    if(m_vaneVertices) delete[] m_vaneVertices;
	if(m_worldP) delete[] m_worldP;
	delete m_rachis;
}

void MlFeather::createNumSegment(short x)
{
    m_numSeg = x;
    m_quilly = new float[m_numSeg];
    m_vaneVertices = new Vector2F[(m_numSeg + 1) * 6];
	m_worldP = new Vector3F[(m_numSeg + 1) * 7];
	m_rachis->create(x);
}

short MlFeather::numSegment() const
{
	return m_numSeg;
}
	
unsigned MlFeather::numVaneVertices() const
{
	return (m_numSeg + 1) * 6;
}
	
unsigned MlFeather::numWorldP() const
{
	return (m_numSeg + 1) * 7;
}

float * MlFeather::quilly()
{
    return m_quilly;
}

float * MlFeather::getQuilly() const
{
     return m_quilly;
}

Vector2F * MlFeather::vane()
{
	return m_vaneVertices;
}

Vector2F * MlFeather::vaneAt(short seg, short side)
{
    return &m_vaneVertices[seg * 6 + 3 * side];
}

Vector2F * MlFeather::getVaneAt(short seg, short side) const
{
    return &m_vaneVertices[seg * 6 + 3 * side];
}

float MlFeather::getLength() const
{
	return m_length;
}

float MlFeather::getWidth(short seg) const
{
	Vector2F * vane = getVaneAt(seg, 0);
	float r = vane->x;
	vane++;
	r += vane->x;
	vane++;
	r += vane->x;

	vane = getVaneAt(seg, 1);
	r = - vane->x;
	vane++;
	r -= vane->x;
	vane++;
	r -= vane->x;
	return r;
}

BoundingRectangle MlFeather::getBoundingRectangle() const
{
	return m_brect;
}

void MlFeather::computeWorldP(const Vector3F & oriPos, const Matrix33F & oriRot, const float& pitch, const float & scale)
{
	
	m_rachis->update(oriPos, oriRot, getLength() * scale, m_skin, pitch);
	
	Vector3F segOrigin = oriPos;
	Matrix33F segSpace = oriRot;
	for(short i = 0; i < m_numSeg; i++) {
		Matrix33F mat = m_rachis->getSpace(i);
		mat.multiply(segSpace);
		
		*segmentOriginWP(i) = segOrigin;
		computeVaneWP(segOrigin, mat, i, 0, scale);
		computeVaneWP(segOrigin, mat, i, 1, scale);
		
		Vector3F d(0.f, 0.f, m_quilly[i] * scale);
		d = mat.transform(d);
		
		segOrigin += d;
		segSpace = mat;
	}
	
	*segmentOriginWP(m_numSeg) = segOrigin;
	computeVaneWP(segOrigin, segSpace, m_numSeg, 0, scale);
	computeVaneWP(segOrigin, segSpace, m_numSeg, 1, scale);
}

Vector3F * MlFeather::segmentOriginWP(short seg)
{
	return &m_worldP[seg * 7];
}

Vector3F * MlFeather::segmentVaneWP(short seg, short side, short idx)
{
	return &m_worldP[seg * 7 + 1 + side * 3 + idx];
}

Vector3F MlFeather::getSegmentOriginWP(short seg) const
{
	return m_worldP[seg * 7];
}

Vector3F MlFeather::getSegmentVaneWP(short seg, short side, short idx) const
{
	return m_worldP[seg * 7 + 1 + side * 3 + idx];
}

void MlFeather::computeVaneWP(const Vector3F & origin, const Matrix33F& space, short seg, short side, float scale)
{
	Vector3F p = origin;
	Vector2F * vane = getVaneAt(seg, side);
	
	const float tapper = getWidth(seg) * -.05f;
	for(short i = 0; i < 3; i++) {
		Vector3F d(tapper * (i + 1), vane->x, vane->y);
		d *= scale;
		d = space.transform(d);
		
		p += d;
		*segmentVaneWP(seg, side, i) = p;
		
		vane++;
	}
}

void MlFeather::setCollision(CollisionRegion * skin)
{
	m_skin = skin;
}

void MlFeather::setFeatherId(short x)
{
	m_id = x;
}
	
short MlFeather::featherId() const
{
	return m_id;
}

void MlFeather::defaultCreate()
{
    createNumSegment(5);
	
    float * quill = quilly();
    quill[0] = 5.f;
    quill[1] = 3.4f;
    quill[2] = 1.9f;
    quill[3] =  .9f;
	quill[4] =  .5f;
    
    Vector2F * vanes = vaneAt(0, 0);
    vanes[0].set(.9f, .9f);
    vanes[1].set(.8f, 1.59f);
    vanes[2].set(.3f, 1.3f);
    vanes = vaneAt(0, 1);
    vanes[0].set(-.6f, .9f);
    vanes[1].set(-.5f, 1.1f);
    vanes[2].set(-.2f, .9f);
    
    vanes = vaneAt(1, 0);
    vanes[0].set(.7f, 1.1f);
    vanes[1].set(.6f, 1.f);
    vanes[2].set(.5f, .9f);
    vanes = vaneAt(1, 1);
    vanes[0].set(-.6f, .62f);
    vanes[1].set(-.6f, .97f);
    vanes[2].set(-.4f, 1.f);
    
    vanes = vaneAt(2, 0);
    vanes[0].set(.4f, .5f);
    vanes[1].set(.5f, .6f);
    vanes[2].set(.4f, .7f);
    vanes = vaneAt(2, 1);
    vanes[0].set(-.3f, .5f);
    vanes[1].set(-.4f, .5f);
    vanes[2].set(-.3f, .7f);
    
    vanes = vaneAt(3, 0);
    vanes[0].set(.4f, .4f);
    vanes[1].set(.3f, .5f);
    vanes[2].set(.3f, .6f);
    vanes = vaneAt(3, 1);
    vanes[0].set(-.3f, .4f);
    vanes[1].set(-.3f, .4f);
    vanes[2].set(-.2f, .6f);
    
    vanes = vaneAt(4, 0);
    vanes[0].set(.1f, .42f);
    vanes[1].set(.1f, .32f);
    vanes[2].set(.1f, .33f);
    vanes = vaneAt(4, 1);
    vanes[0].set(-.1f, .42f);
    vanes[1].set(-.1f, .32f);
    vanes[2].set(-.1f, .33f);
	
	vanes = vaneAt(5, 0);
    vanes[0].set(.01f, .42f);
    vanes[1].set(.01f, .32f);
    vanes[2].set(.01f, .33f);
    vanes = vaneAt(5, 1);
    vanes[0].set(-.01f, .42f);
    vanes[1].set(-.01f, .32f);
    vanes[2].set(-.01f, .33f);
	
	computeBounding();
	computeLength();
}

void MlFeather::computeLength()
{
	m_length = 0.f;
	for(short i=0; i < m_numSeg; i++)
		m_length += m_quilly[i];
	m_rachis->computeAngles(m_quilly, m_length);
}

void MlFeather::computeBounding()
{
	m_brect.reset();
	Vector2F c = m_uv;
	Vector2F p;
	for(short i = 0; i <= m_numSeg; i++) {
		m_brect.update(c);
		
		Vector2F* vane = getVaneAt(i, 0);
		
		p = c;
		p += vane[0];
		p += vane[1];
		p += vane[2];
		m_brect.update(p);
		
		vane = getVaneAt(i, 1);
		
		p = c;
		p += vane[0];
		p += vane[1];
		p += vane[2];
		m_brect.update(p);
		
		if(i < m_numSeg)
			c += Vector2F(0.f, getQuilly()[i]);
	}
	m_brect.update(c);
}

Vector2F MlFeather::baseUV() const
{
	return m_uv;
}

void MlFeather::setBaseUV(const Vector2F & d)
{
	m_uv = d;
}

void MlFeather::translateUV(const Vector2F & d)
{
	m_uv += d;
	m_brect.translate(d);
}

void MlFeather::verbose()
{
	std::cout<<"feather status:\n id "<<featherId();
	std::cout<<"\n n segment "<<numSegment();
	std::cout<<"\n length "<<getLength();
	std::cout<<"\n base uv ("<<m_uv.x<<","<<m_uv.y<<")\n";
}
