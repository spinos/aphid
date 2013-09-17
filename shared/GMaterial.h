/*
 *  GMaterial.h
 *  mallard
 *
 *  Created by jian zhang on 9/18/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef G_MATERIAL_H
#define G_MATERIAL_H

#ifdef WIN32
#include <gExtension.h>
#else
#include <gl_heads.h>
#endif

#include <AllMath.h>

class GMaterial {
public:
	GMaterial();		
	GMaterial( Color4 ambient,
             Color4 diffuse,
             Color4 specular,
             Color4 emission,
             float shininess);
 
    void apply() const;
 
    Color4 m_Ambient;
    Color4 m_Diffuse;
    Color4 m_Specular;
    Color4 m_Emission;
    float  m_Shininess;
};

#endif