/// https://www.opengl.org/wiki/Vertex_Rendering#Instancing
/// http://www.informit.com/articles/article.aspx?p=2033340&seqNum=5
#include "GlslInstancer.h"

namespace aphid {
    
GlslInstancer::GlslInstancer()
{}

GlslInstancer::~GlslInstancer()
{}

const char* GlslInstancer::vertexProgramSource() const
{
	return "uniform mat4 ModelViewMatrix;"
"uniform mat4 ProjectionMatrix;"
// "layout(location = 0) in vec3 in_position;"
"void main()"
"{"
"		gl_Position = ftransform();"
"		gl_FrontColor = gl_Color;"
"}";
}

const char* GlslInstancer::fragmentProgramSource() const
{
	return "void main()"
"{"
"		gl_FragColor = gl_Color * vec4 (0.99);"
"}";
}

}
