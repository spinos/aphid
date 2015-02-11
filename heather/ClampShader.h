/*
 *  ClampShader.h
 *  heather
 *
 *  Created by jian zhang on 2/12/15.
 *  Copyright 2015 __MyCompanyName__. All rights reserved.
 *
 */

#include <GlslBase.h>

class ClampShader : public GLSLBase {
public:
	ClampShader();
	virtual ~ClampShader();
	void setTextures(const GLuint & bgdZImg, const GLuint & bgdCImg, 
				const GLuint & depthImg, 
				const GLuint & colorImg);
	void setClippings(const float & nearClipping, const float & farClipping);
protected:
	virtual const char* vertexProgramSource() const;
	virtual const char* fragmentProgramSource() const;
	virtual void updateShaderParameters() const;
private:
	GLuint m_bgdZImg, m_bgdCImg, m_depthImg, m_colorImg;
	float m_nearClipping, m_farClipping;
};