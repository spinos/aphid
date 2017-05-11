#include "DrawGrid2.h"
#include <math/BoundingBox.h>
#include <gl_heads.h>

namespace aphid {

DrawGrid2::DrawGrid2() : 
m_numVertices(0)
{}
    
DrawGrid2::~DrawGrid2()
{} 

void DrawGrid2::setBoxFace(float * pos,
                    float * mnl,
                    const BoundingBox & bx,
                    const int & iface)
{
    const Vector3F pmn = bx.getMin();
    const Vector3F pmx = bx.getMax();
    Vector3F vert[4];
    Vector3F nmlf;
    switch(iface) {
    case 0:
///      3
///    / |
///  2   0
///  | /
///  1    
            vert[0].set(pmn.x, pmn.y, pmn.z);
            vert[1].set(pmn.x, pmn.y, pmx.z);
            vert[2].set(pmn.x, pmx.y, pmx.z);
            vert[3].set(pmn.x, pmx.y, pmn.z);
            nmlf.set(-1.f, 0.f, 0.f);
        break;
        case 1:
///      3
///    / |
///  0   2
///  | /
///  1  
            vert[0].set(pmx.x, pmx.y, pmx.z);
            vert[1].set(pmx.x, pmn.y, pmx.z);
            vert[2].set(pmx.x, pmn.y, pmn.z);
            vert[3].set(pmx.x, pmx.y, pmn.z);
            nmlf.set(1.f, 0.f, 0.f);
        break;
        case 2:            
///    0 -  1
///   /    /
///  3 -  2
            vert[0].set(pmn.x, pmn.y, pmn.z);
            vert[1].set(pmx.x, pmn.y, pmn.z);
            vert[2].set(pmx.x, pmn.y, pmx.z);
            vert[3].set(pmn.x, pmn.y, pmx.z);
            nmlf.set(0.f, -1.f, 0.f);
        break;
        case 3:
///    2 -  1
///   /    /
///  3 -  0
            vert[0].set(pmx.x, pmx.y, pmx.z);
            vert[1].set(pmx.x, pmx.y, pmn.z);
            vert[2].set(pmn.x, pmx.y, pmn.z);
            vert[3].set(pmn.x, pmx.y, pmx.z);
            nmlf.set(0.f, 1.f, 0.f);
        break;
        case 4:
///  1 - 2
///  |   |
///  0 - 3
            vert[0].set(pmn.x, pmn.y, pmn.z);
            vert[1].set(pmn.x, pmx.y, pmn.z);
            vert[2].set(pmx.x, pmx.y, pmn.z);
            vert[3].set(pmx.x, pmn.y, pmn.z);
            nmlf.set(0.f, 0.f, -1.f);
        break;
    default:
///  1 - 0
///  |   |
///  2 - 3
            vert[0].set(pmx.x, pmx.y, pmx.z);
            vert[1].set(pmn.x, pmx.y, pmx.z);
            vert[2].set(pmn.x, pmn.y, pmx.z);
            vert[3].set(pmx.x, pmn.y, pmx.z);
            nmlf.set(0.f, 0.f, 1.f);
        ;
    }
    
    memcpy(pos, &vert[0], 12);
    memcpy(&pos[3], &vert[1], 12);
    memcpy(&pos[6], &vert[2], 12);
    memcpy(&pos[9], &vert[0], 12);
    memcpy(&pos[12], &vert[2], 12);
    memcpy(&pos[15], &vert[3], 12);
    
    for(int i = 0; i<6;++i ) {
        memcpy(&mnl[i*3], &nmlf, 12);
    }
}

void DrawGrid2::setUniformColor(const float * col)
{
    for(int i = 0; i<m_numVertices;++i ) {
        memcpy(&m_vertexColors[i*3], col, 12);
    }
}

void DrawGrid2::setFaceColor(float * dst,
					const float * col)
{
	for(int i = 0; i<6;++i ) {
        memcpy(&dst[i*3], col, 12);
    }
}

void DrawGrid2::drawSolidGrid() const
{
	glColorPointer(3, GL_FLOAT, 0, (const GLfloat*)m_vertexColors.get());
	glNormalPointer(GL_FLOAT, 0, (const GLfloat*)m_vertexNormals.get());
	glVertexPointer(3, GL_FLOAT, 0, (const GLfloat*)m_vertexPoints.get());

	glDrawArrays(GL_TRIANGLES, 0, m_numVertices);
}

static const float sOctaVertOffset[24][3] = {
{ 1.f, 0.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 0.f, 1.f},/// 0
{ 1.f, 0.f, 0.f},
{ 0.f, 0.f,-1.f},
{ 0.f, 1.f, 0.f},/// 1
{-1.f, 0.f, 0.f},
{ 0.f, 1.f, 0.f},
{ 0.f, 0.f,-1.f},/// 2
{-1.f, 0.f, 0.f},
{ 0.f, 0.f, 1.f},
{ 0.f, 1.f, 0.f},/// 3
{ 1.f, 0.f, 0.f},
{ 0.f, 0.f, 1.f},
{ 0.f,-1.f, 0.f},/// 4
{ 1.f, 0.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f, 0.f,-1.f},/// 5
{-1.f, 0.f, 0.f},
{ 0.f, 0.f,-1.f},
{ 0.f,-1.f, 0.f},/// 6
{-1.f, 0.f, 0.f},
{ 0.f,-1.f, 0.f},
{ 0.f, 0.f, 1.f},/// 7
};

void DrawGrid2::setOctahedron(float * pos,
                    float * mnl,
					const Vector3F & pncen,
					const Vector3F & pnnml,
					const float & pnwd)
{
	Vector3F vert;
	for(int i=0;i<24;++i) {
		vert = pncen + Vector3F(sOctaVertOffset[i][0] * pnwd,
							sOctaVertOffset[i][1] * pnwd,
							sOctaVertOffset[i][2] * pnwd);
		pos[i*3] = vert.x;
		pos[i*3+1] = vert.y;
		pos[i*3+2] = vert.z;
		mnl[i*3] = pnnml.x;
		mnl[i*3+1] = pnnml.y;
		mnl[i*3+2] = pnnml.z;
	}
}

}
