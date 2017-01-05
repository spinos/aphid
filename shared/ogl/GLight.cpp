/*
 *  GLight.h
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "GLight.h"

namespace aphid {

GLight::GLight() 
{
	m_LightID = GL_LIGHT0;
	m_Ambient = Color4( 0.2, 0.2, 0.2, 1.0 );
	m_Diffuse = Color4( .89, .89, .9, 1.0 );
	m_Specular = Color4( .8, .8, .9, 1.0 );
	m_Position = Float4( 100.0, 100.0, 100.0, 1.0 );
}
		
GLight::GLight(GLenum LightID, 
		Color4 Ambient, 
		Color4 Diffuse,
		Color4 Specular,
		Float4 Position)
{
	m_LightID = LightID;
	m_Ambient = Ambient;
	m_Diffuse = Diffuse;
	m_Specular = Specular;
	m_Position = Position;
}

void GLight::activate() const
{
	glEnable(GL_LIGHTING);
	glEnable( m_LightID );
	glLightfv( m_LightID, GL_AMBIENT, &(m_Ambient.r) );
	glLightfv( m_LightID, GL_DIFFUSE, &(m_Diffuse.r) );
	glLightfv( m_LightID, GL_SPECULAR, &(m_Specular.r) );
	glLightfv( m_LightID, GL_POSITION, &(m_Position.x) );
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	
}

void GLight::deactivate() const
{
	glDisable( m_LightID );
}
 
}   