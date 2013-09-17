/*
 *  GMaterial.h
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <GMaterial.h>

GMaterial::GMaterial() 
{
	m_Ambient = Color4(0.1, 0.1, 0.1, 1.0);
	m_Diffuse = Color4(0.8, 0.8, 0.8, 1.0);
	m_Specular = Color4(0.0, 0.0, 0.0, 1.0);
	m_Emission = Color4(0.0, 0.0, 0.0, 1.0);
	m_Shininess = 10.f;
}
	
GMaterial::GMaterial( Color4 ambient,
		 Color4 diffuse,
		 Color4 specular,
		 Color4 emission,
		 float shininess)
{
	m_Ambient = ambient;
	m_Diffuse = diffuse;
	m_Specular = specular;
	m_Emission = emission;
	m_Shininess = shininess;
}

void GMaterial::apply() const
{
	glMaterialfv( GL_FRONT_AND_BACK, GL_AMBIENT, &(m_Ambient.r) );
	glMaterialfv( GL_FRONT_AND_BACK, GL_DIFFUSE, &(m_Diffuse.r) );
	glMaterialfv( GL_FRONT_AND_BACK, GL_SPECULAR, &(m_Specular.r) );
	glMaterialfv( GL_FRONT_AND_BACK, GL_EMISSION, &(m_Emission.r) );
	glMaterialf( GL_FRONT_AND_BACK, GL_SHININESS, m_Shininess );
}
