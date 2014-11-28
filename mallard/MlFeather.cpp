#include "MlFeather.h"
#include "MlRachis.h"
#include "MlVane.h"
#include <CollisionRegion.h>
#include <AdaptableStripeBuffer.h>
#include <MlTessellate.h>
MlFeather::MlFeather()
{
	m_rachis = new MlRachis;
	m_vane = new MlVane[2];
	m_tessel = new MlTessellate; 
	simpleCreate();
	
	setSeed(1);
	setNumSeparate(2);
	setSeparateStrength(0.f);
	setFuzzy(0.f);
	m_scale = 1.f;
}

MlFeather::~MlFeather() 
{
	delete m_rachis;
	delete[] m_vane;
	delete m_tessel;
}

void MlFeather::createNumSegment(short x)
{
	BaseFeather::createNumSegment(x);
	m_rachis->create(x);
	m_tessel->setFeather(this);
}

void MlFeather::createVanes()
{
	DeformableFeather::createVanes();
	m_vane[0].copy(*uvVane(0));
	m_vane[1].copy(*uvVane(1));
}

void MlFeather::bend()
{
	m_rachis->bend();
}

void MlFeather::bendAt(unsigned faceIdx, float patchU, float patchV, const Vector3F & oriPos, const Matrix33F & oriRot, const float & scale)
{
	m_rachis->bend(faceIdx, patchU, patchV, oriPos, oriRot, scale * shaftLength(), m_skin);
}

void MlFeather::pitchAndCurl(const float & p, const float & c)
{
	m_rachis->pitchAndCurl(p, c);
}

void MlFeather::computeWorldP(const Vector3F & oriPos, const Matrix33F & oriRot, const float & scaleLength, const float & scaleWidth)
{
	m_scale = scaleLength;
	const float xscale = m_scale * 32.f;
	Vector3F segOrigin = oriPos;
	Matrix33F segSpace = oriRot;
	const short numSeg = numSegment();
	for(short i = 0; i < numSeg; i++) {
		Matrix33F mat = m_rachis->getSpace(i);
		mat.multiply(segSpace);
		
		normal(i)->set(mat.M(0, 0), mat.M(0, 1), mat.M(0, 2));
		
		computeVaneWP(segOrigin, mat, i, xscale, scaleWidth);

		Vector3F d(0.f, 0.f, quilly()[i] * m_scale);
		d = mat.transform(d);
		
		segOrigin += d;
		segSpace = mat;
	}
}

Vector3F * MlFeather::segmentVaneWP(short u, short v, short side)
{
	return m_vane[side].railCV(u, v);
}

void MlFeather::computeVaneWP(const Vector3F & origin, const Matrix33F& space, short seg, const float & xscale, const float & scaleWidth)
{
	Vector3F d;
	for(short i=0; i < numBind(seg); i++) {
		DeformableFeather::BindCoord *bind = getBind(seg, i);
		d.set(- 0.001f * bind->_taper, bind->_objP.x * scaleWidth, bind->_objP.y);
		d *= xscale;
		d = space.transform(d);
		d = origin + d;
		*segmentVaneWP(bind->_u,  bind->_v,  bind->_rgt) = d;
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
	BaseFeather::simpleCreate(ns);
}

void MlFeather::computeLength()
{
	BaseFeather::computeLength();
	m_rachis->computeLengths(quilly(), shaftLength());
}

void MlFeather::changeNumSegment(int d)
{
	BaseFeather::changeNumSegment(d);
}

void MlFeather::getBoundingBox(BoundingBox & box)
{
	
	for(short i = 0; i <= numSegment(); i++) {
		box.update(*segmentVaneWP(i, 3, 0));
		box.update(*segmentVaneWP(i, 1, 0));
	}
}

Float2 * MlFeather::angles() const
{
	return m_rachis->angles();
}

float MlFeather::bendDirection() const
{
	return m_rachis->bendDirection();
}

MlVane * MlFeather::vane(short side) const
{
	return &m_vane[side];
}

void MlFeather::verbose()
{
	std::cout<<"feather index:\n id "<<featherId();
	std::cout<<"\n n segment "<<numSegment();
	std::cout<<"\n length "<<shaftLength();
	std::cout<<"\n base uv ("<<baseUV().x<<","<<baseUV().y<<")";
	std::cout<<"\n shaft shrink "<<m_shaftShrink;
	std::cout<<"\n barb shrink "<<m_barbShrink<<"\n";
}

void MlFeather::samplePosition(Vector3F * dst)
{
	const unsigned gridU = resShaft();
	const unsigned gridV = resBarb();
	const float du = 1.f/(float)gridU;
	const float dv = 1.f/(float)gridV;
	
	
	unsigned acc = 0;
	for(unsigned i = 0; i <= gridU; i++) {
		m_vane[0].setU(du*i);
		for(unsigned j = 0; j <= gridV; j++) {
			m_vane[0].pointOnVane(dv * j, dst[acc]);
			acc++;
		}
		m_vane[0].modifyLength(du*i, gridV, &dst[acc - gridV - 1], 1.f);
	}
	
	
	for(unsigned i = 0; i <= gridU; i++) {
		m_vane[1].setU(du*i);
		for(unsigned j = 0; j <= gridV; j++) {
			m_vane[1].pointOnVane(dv * j, dst[acc]);
			acc++;
		}
		m_vane[1].modifyLength(du*i, gridV, &dst[acc - gridV - 1], 1.f);
	}
}

void MlFeather::setSeed(unsigned s)
{
	m_vane[0].setSeed(s);
	m_vane[1].setSeed(s + 64873);
}

void MlFeather::setNumSeparate(unsigned n)
{
	m_vane[0].setNumSparate(n);
	m_vane[1].setNumSparate(n);
	m_numSeparate = n;
}

void MlFeather::setSeparateStrength(float k)
{
	m_vane[0].setSeparateStrength(k);
	m_vane[1].setSeparateStrength(k);
	m_separateStrength = k;
}

void MlFeather::setFuzzy(float f)
{
	m_vane[0].setFuzzy(f);
	m_vane[1].setFuzzy(f);
	m_fuzzy = f;
}

unsigned MlFeather::seed() const
{
	return m_vane[0].seed();
}

unsigned MlFeather::numSeparate() const
{
	return m_numSeparate;
}

float MlFeather::fuzzy() const
{
	return m_fuzzy;
}

float MlFeather::separateStrength() const
{
	return m_separateStrength;
}

void MlFeather::testVane()
{
	Vector3F oriP(4.f, -2.f, 4.f);
	Matrix33F oriR; oriR.fill(Vector3F::ZAxis, Vector3F::XAxis, Vector3F::YAxis);
	computeWorldP(oriP, oriR, 2.f, 1.f);
	separateVane();
}

void MlFeather::separateVane()
{
	m_vane[0].separate();
	m_vane[1].separate();
}

void MlFeather::computeNoise()
{
	m_vane[0].computeNoise();
	m_vane[1].computeNoise();
}

void MlFeather::samplePosition(float lod)
{
	unsigned nu, nv;
	computeLODGrid(lod, nu, nv);
	stripe()->begin();
	
	samplePosition(nu, nv, 0, lod);
	samplePosition(nu, nv, 1, lod);
}

void MlFeather::samplePosition(unsigned nu, unsigned nv, int side, float lod)
{
	float rootWidth = scaledShaftLength();
	if(type() == 1) rootWidth = quilly()[0] * m_scale;
	
	rootWidth *= m_barbWidthScale / (float)nu;

	float tipWidth = rootWidth * (1.f - m_barbShrink);

	const float du = 1.f/(float)nu;
	const float dv = 1.f/(float)nv;
	float paramv, paramu, tapering;
	
	for(unsigned i = 0; i < nu; i++) {
		*stripe()->currentNumCvs() = nv + 1;
		
		Vector3F * coord = stripe()->currentPos();
		float * w = stripe()->currentWidth();
		
		paramu = (float)i/(float)nu;
		if(type() > 0) paramu = 1.f - paramu;
		
		tapering = 1.f - paramu * m_shaftShrink;

		m_vane[side].setU(du*i);
		
		for(unsigned j = 0; j <= nv; j++) {
			m_vane[side].pointOnVane(dv * j, coord[j]);
			paramv = (float)j / (float)nv;
			w[j] = rootWidth * (1.f - paramv) + tipWidth * paramv;
			w[j] *= tapering * .5f;
		}
		
		m_vane[side].modifyLength(du*i, nv + 2, coord, lod);
		
		stripe()->next();
	}
}

float MlFeather::scaledShaftLength() const
{
	return m_scale * shaftLength();
}

Vector3F * MlFeather::patchCenterP(short seg)
{
	if(type() == 0) return m_vane[0].railCV(seg, 0);
	return m_vane[0].railCV(0, seg);
}

Vector3F * MlFeather::patchWingP(short seg, short side)
{
	if(type() == 0) return m_vane[side].railCV(seg, 3);
	return m_vane[side].railCV(3, seg);
}

MlTessellate * MlFeather::tessellator() const
{
	return m_tessel;
}

void MlFeather::dumpWP(Vector3F * p, float * u, float * v, unsigned & cp)
{
    Vector3F uvc;
    for(short i = 0; i <= numSegment(); i++) {
        p[cp] = *patchCenterP(i);
        uvc = *patchCenterUV(i);
        u[cp] = uvc.x;
        v[cp] = uvc.y;
        cp++;
        p[cp] = *patchWingP(i, 0);
        uvc = *patchWingUV(i, 0);
        u[cp] = uvc.x;
        v[cp] = uvc.y;
        cp++;
        p[cp] = *patchWingP(i, 1);
        uvc = *patchWingUV(i, 1);
        u[cp] = uvc.x;
        v[cp] = uvc.y;
        cp++;
	}
}

void MlFeather::dumpFV(unsigned * fv, unsigned & cfv)
{
    for(short i = 0; i < numSegment(); i++) {
        fv[cfv] = i * 3;
        cfv++;
        fv[cfv] = i * 3 + 1;
        cfv++;
        fv[cfv] = (i + 1) * 3 + 1;
        cfv++;
        fv[cfv] = (i + 1) * 3;
        cfv++;
        
        fv[cfv] = i * 3;
        cfv++;
        fv[cfv] = (i + 1) * 3;
        cfv++;
        fv[cfv] = (i + 1) * 3 + 2;
        cfv++;
        fv[cfv] = i * 3 + 2;
        cfv++;
    }
}
