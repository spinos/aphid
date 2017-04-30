#ifndef GLFRAMEBUFFER_H
#define GLFRAMEBUFFER_H
#include <gl_heads.h>
class GlFramebuffer
{
public:
	GlFramebuffer(int w, int h);
	virtual ~GlFramebuffer();
	const bool hasFbo() const;
	void begin() const;
	void end() const;
	const GLuint colorTexture() const;
private:	
	GLuint fbo;
	GLuint depthBuffer;
	GLuint img;
	int m_framebufferWidth, m_framebufferHeight;
	bool fHasFbo;
};
#endif        //  #ifndef GLFRAMEBUFFER_H

