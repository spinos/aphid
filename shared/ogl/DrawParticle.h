/*
 *  DrawParticle.h
 *  
 *
 *  Created by jian zhang on 1/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_PARTICLE_H
#define APH_OGL_DRAW_PARTICLE_H

#include <ogl/DrawGlyph.h>
#include <boost/scoped_array.hpp>

namespace aphid {

struct Float4;
class GlslLegacyInstancer;

class DrawParticle : public DrawGlyph {

	GlslLegacyInstancer * m_instancer;
/// transform and color
	boost::scoped_array<Float4 > m_particles;
	int m_numParticles;
	
public:
	DrawParticle();
	virtual ~DrawParticle();
	
protected:
	int initGlsl();
	
	void createParticles(int np);
	
	const int & numParticles() const;
	
	void drawParticles() const;
	
/// i-th particle in rows
/// rx0 ry0 rz0 tx
/// rx1 ry1 rz1 ty
/// rx2 ry2 rz2 tz
/// cr cg cb ca 
	Float4 * particleR(int i);
	const Float4 * particleR(int i) const;
	
	void permutateParticleColors();
	
};

}
#endif