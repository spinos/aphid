/*
 *  GMaterial.h
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include "GMaterial.h"
#include <gl_heads.h>

namespace aphid {

GMaterial::GMaterial() 
{
	m_Ambient = Color4(0.1, 0.1, 0.1, 1.0);
	m_Diffuse = Color4(0.8, 0.8, 0.8, 1.0);
	m_backDiffuse = Color4(1.0, 0.0, 0.0, 1.0);
	m_Specular = Color4(0.0, 0.0, 0.0, 1.0);
	m_Emission = Color4(0.0, 0.0, 0.0, 1.0);
	m_Shininess = 10.f;
}
	
GMaterial::GMaterial( Color4 ambient,
		 Color4 frontDiffuse,
			Color4 backDiffuse,
		 Color4 specular,
		 Color4 emission,
		 float shininess)
{
	m_Ambient = ambient;
	m_Diffuse = frontDiffuse;
	m_backDiffuse = backDiffuse;
	m_Specular = specular;
	m_Emission = emission;
	m_Shininess = shininess;
}

void GMaterial::apply() const
{
	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, &(m_Ambient.r) );
	glMaterialfv( GL_FRONT, GL_DIFFUSE, &(m_Diffuse.r) );
	glMaterialfv( GL_BACK, GL_DIFFUSE, &(m_backDiffuse.r) );
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, &(m_Specular.r) );
	glMaterialfv( GL_FRONT_AND_BACK, GL_EMISSION, &(m_Emission.r) );
	glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, m_Shininess );
}

}