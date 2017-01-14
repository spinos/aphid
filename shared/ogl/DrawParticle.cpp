/*
 *  DrawParticle.cpp
 *  
 *
 *  Created by jian zhang on 1/14/17.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#include "DrawParticle.h"
#include <math/ATypes.h>
#include <ogl/GlslInstancer.h>
#include <math/miscfuncs.h>

namespace aphid {

DrawParticle::DrawParticle()
{
	m_numParticles = 0;
	m_instancer = new GlslLegacyInstancer;
}
	
DrawParticle::~DrawParticle()
{}

bool DrawParticle::initGlsl()
{
	std::string diaglog;
    m_instancer->diagnose(diaglog);
    std::cout<<diaglog;
    m_instancer->initializeShaders(diaglog);
    std::cout<<diaglog;
    std::cout.flush();
    return m_instancer->isDiagnosed();
}

void DrawParticle::createParticles(int np)
{
	m_numParticles = np;
	m_particles.reset(new Float4[np<<2]);
}
	
const int & DrawParticle::numParticles() const
{ return m_numParticles; }

void DrawParticle::drawParticles() const
{
	m_instancer->programBegin();

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
        
	for(int i=0;i<m_numParticles;++i) {
	    const Float4 *d = particleR(i);
	    glMultiTexCoord4fv(GL_TEXTURE1, (const float *)d);
	    glMultiTexCoord4fv(GL_TEXTURE2, (const float *)&d[1]);
	    glMultiTexCoord4fv(GL_TEXTURE3, (const float *)&d[2]);
		m_instancer->setDiffueColorVec((const float *)&d[3]);
	    
	    drawAGlyph();
        
	}
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	m_instancer->programEnd();
}

Float4 * DrawParticle::particleR(int i)
{ return &m_particles[i<<2]; }

const Float4 * DrawParticle::particleR(int i) const
{ return &m_particles[i<<2]; }

void DrawParticle::permutateParticleColors()
{
	static const float permt[8] = {0.05f, .45f,
	.2f, .7f, .135f, .325f, .655f, .895f};
	
	for(int i=0;i<m_numParticles;++i) {
		Float4 * pr = particleR(i);
        pr[3] = Float4(permt[rand() & 7],
						permt[rand() & 7],
						permt[rand() & 7],
						1);
    }
}

}

