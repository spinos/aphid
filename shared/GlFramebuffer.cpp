#include "GlFramebuffer.h"

GlFramebuffer::GlFramebuffer(int w, int h)
{
    m_framebufferWidth = w; 
    m_framebufferHeight = h;
	glGenFramebuffersEXT(1, &fbo);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

// Create the render buffer for depth	
	glGenRenderbuffersEXT(1, &depthBuffer);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depthBuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT24, m_framebufferWidth, m_framebufferHeight);

// Now setup the first texture to render to
	glGenTextures(1, &img);
	glBindTexture(GL_TEXTURE_2D, img);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, m_framebufferWidth, m_framebufferHeight, 0, GL_RGBA, GL_HALF_FLOAT_ARB, 0);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
// And attach it to the FBO so we can render to it
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, img, 0);
	
// Attach the depth render buffer to the FBO as it's depth attachment
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depthBuffer);

	GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	if(status != GL_FRAMEBUFFER_COMPLETE_EXT) {
		fHasFbo = 0;
	}
	
// Unbind the FBO for now
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	fHasFbo = 1;
}

GlFramebuffer::~GlFramebuffer() {}

const bool GlFramebuffer::hasFbo() const
{ return fHasFbo; }

void GlFramebuffer::begin() const
{
// First we bind the FBO so we can render to it
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

// Save the view port and set it to the size of the texture
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	glViewport(0,0, m_framebufferWidth, m_framebufferHeight);
	
// Then render as normal
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glClearDepth(1.0f);		  
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer

}

void GlFramebuffer::end() const
{
	glPopAttrib();
	// glReadPixels( 0, 0, m_framebufferWidth, m_framebufferHeight, GL_RGBA, GL_FLOAT, fPixels);
// unbind fbo
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

const GLuint GlFramebuffer::colorTexture() const
{ return img; }
