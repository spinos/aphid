#include "MlFeather.h"
#include "MlRachis.h"
#include <CollisionRegion.h>
MlFeather::MlFeather() : m_quilly(0), m_vaneVertices(0), m_worldP(0)
{
	m_rachis = new MlRachis;
	m_uv.set(4.f, 4.f);
	simpleCreate();
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
	if(m_quilly) delete[] m_quilly;
    if(m_vaneVertices) delete[] m_vaneVertices;
	if(m_worldP) delete[] m_worldP;
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
	
	m_rachis->update(oriPos, oriRot, scale * getLength(), m_skin, pitch);
	
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
	
	const float tapper = getWidth(seg) * -.01f;
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

void MlFeather::simpleCreate(int ns)
{
    createNumSegment(ns);
	
    float * quill = quilly();
	int i;
	for(i = 0; i < ns; i++) {
		if(i < ns - 2)
			quill[i] = 3.f;
		else if(i < ns - 1)
			quill[i] = 1.7f;
		else
			quill[i] = .8f;
    }
	
	Vector2F * vanesR;
	Vector2F * vanesL;
	for(i = 0; i <= ns; i++) {
		vanesR = vaneAt(i, 0);
		vanesL = vaneAt(i, 1);
		
		if(i < ns - 2) {
			vanesR[0].set(.9f, .8f);
			vanesR[1].set(.8f, 1.1f);
			vanesR[2].set(.2f, 1.3f);
			
			vanesL[0].set(-.9f, .8f);
			vanesL[1].set(-.8f, 1.1f);
			vanesL[2].set(-.2f, 1.3f);
		}
		else if(i < ns - 1) {
			vanesR[0].set(.6f, .6f);
			vanesR[1].set(.4f, .5f);
			vanesR[2].set(.05f, .6f);
			
			vanesL[0].set(-.6f, .6f);
			vanesL[1].set(-.4f, .5f);
			vanesL[2].set(-.05f, .6f);
		}
		else if(i < ns) {
			vanesR[0].set(.3f, .3f);
			vanesR[1].set(.2f, .3f);
			vanesR[2].set(0.f, .4f);
			
			vanesL[0].set(-.3f, .3f);
			vanesL[1].set(-.2f, .3f);
			vanesL[2].set(0.f, .4f);
		}
		else {
			vanesR[0].set(0.f, .2f);
			vanesR[1].set(0.f, .1f);
			vanesR[2].set(0.f, .1f);
			
			vanesL[0].set(0.f, .2f);
			vanesL[1].set(0.f, .1f);
			vanesL[2].set(0.f, .1f);
		}
	}
	
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

float* MlFeather::selectVertexInUV(const Vector2F & p, bool & yOnly, Vector2F & wp)
{
	float * r = 0;
	float minD = 10e8;
	yOnly = true;
	
	Vector2F puv = m_uv;
	float *q = quilly();
	int i, j;
	for(i=0; i < numSegment(); i++) {
		puv += Vector2F(0.f, *q);
		
		if(p.distantTo(puv) < minD) {
			minD = p.distantTo(puv);
			r = q;
			wp = puv;
		}
		
		q++;
	}
	
	q = quilly();
	puv = m_uv;
	
	Vector2F pvane;
	for(i=0; i <= numSegment(); i++) {
		
		pvane = puv;
		Vector2F * vanes = vaneAt(i, 0);
		
		for(j = 0; j < 3; j++) {
			pvane += *vanes;
			if(p.distantTo(pvane) < minD) {
				minD = p.distantTo(pvane);
				r = (float *)vanes;
				yOnly = false;
				wp = pvane;
			}
			vanes++;
		}

		pvane = puv;
		vanes = getVaneAt(i, 1);
		
		for(j = 0; j < 3; j++) {
			pvane += *vanes;
			if(p.distantTo(pvane) < minD) {
				minD = p.distantTo(pvane);
				r = (float *)vanes;
				yOnly = false;
				wp = pvane;
			}
			vanes++;
		}
		
		if(i < numSegment()) {
			puv += Vector2F(0.f, *q);
			q++;
		}
	}
	
	return r;
}

void MlFeather::changeNumSegment(int d)
{
	float * bakQuilly = new float[m_numSeg];
    Vector2F *bakVaneVertices = new Vector2F[(m_numSeg + 1) * 6];
	
	int i, j;
	for(i = 0; i < m_numSeg; i++)
		bakQuilly[i] = quilly()[i];
		
	for(i = 0; i < (m_numSeg + 1) * 6; i++)
		bakVaneVertices[i] = vane()[i];
		
	createNumSegment(m_numSeg + d);
	
	if(d > 0) {
		for(i = 0; i < m_numSeg; i++) {
			if(i == 0) quilly()[i] = bakQuilly[0];
			else quilly()[i] = bakQuilly[i - 1];
		}
		for(i = 0; i <= m_numSeg; i++) {
			if(i == 0) {
				for(j = 0; j < 6; j++)
					vane()[i * 6 + j] = bakVaneVertices[j] ;
			}
			else {
				for(j = 0; j < 6; j++)
					vane()[i * 6 + j] = bakVaneVertices[(i - 1) * 6 + j] ;
			}
		}
	}
	else {
		for(i = 0; i < m_numSeg; i++) {
			if(i < m_numSeg -1) quilly()[i] = bakQuilly[i];
			else quilly()[i] = bakQuilly[i + 1];
		}
		for(i = 0; i <= m_numSeg; i++) {
			if(i < m_numSeg -1) {
				for(j = 0; j < 6; j++)
					vane()[i * 6 + j] = bakVaneVertices[i * 6 + j] ;
			}
			else {
				for(j = 0; j < 6; j++)
					vane()[i * 6 + j] = bakVaneVertices[(i + 1) * 6 + j] ;
			}
		}
	}
	
	delete[] bakQuilly;
	delete[] bakVaneVertices;
	
	computeBounding();
	computeLength();
}

void MlFeather::verbose()
{
	std::cout<<"feather status:\n id "<<featherId();
	std::cout<<"\n n segment "<<numSegment();
	std::cout<<"\n length "<<getLength();
	std::cout<<"\n base uv ("<<m_uv.x<<","<<m_uv.y<<")\n";
}
