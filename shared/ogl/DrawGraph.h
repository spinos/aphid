/*
 *  DrawGraph.h
 *  
 *  Tn as node type
 *  Te as edge type
 *  Created by zhang on 17-1-30.
 *  Copyright 2017 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef APH_OGL_DRAW_GRAPH_H
#define APH_OGL_DRAW_GRAPH_H

#include <ogl/DrawGlyph.h>
#include <ogl/GlslInstancer.h>
#include <gl_heads.h>

namespace aphid {

template<typename Tn, typename Te>
class DrawGraph : public DrawGlyph {

    GlslLegacyInstancer * m_instancer;
    float m_nodeSize;
    
public:
    DrawGraph();
    virtual ~DrawGraph();
    
    bool initGlsl();
    
    void setDrawNodeSize(const float & x);
    
    void drawEdge(const AGraph<Tn, Te > * gph);
    void drawNode(const AGraph<Tn, Te > * gph);
    
protected:
    void getNodeColor(float * col, const float & val);
    
private:

};

template<typename Tn, typename Te>
DrawGraph<Tn, Te>::DrawGraph()
{ 
    m_instancer = new GlslLegacyInstancer; 
    m_nodeSize = .25f;
}

template<typename Tn, typename Te>
DrawGraph<Tn, Te>::~DrawGraph()
{}

template<typename Tn, typename Te>
void DrawGraph<Tn, Te>::setDrawNodeSize(const float & x)
{ m_nodeSize = x; }

template<typename Tn, typename Te>
bool DrawGraph<Tn, Te>::initGlsl()
{
	std::string diaglog;
    m_instancer->diagnose(diaglog);
    std::cout<<diaglog;
    m_instancer->initializeShaders(diaglog);
    std::cout<<diaglog;
    std::cout.flush();
    return m_instancer->isDiagnosed();
}

template<typename Tn, typename Te>
void DrawGraph<Tn, Te>::drawEdge(const AGraph<Tn, Te > * gph)
{
    const Te * egs = gph->edges();
    const int & ne = gph->numEdges();
    const Tn * ns = gph->nodes();
    
    glBegin(GL_LINES);
    for(int i=0;i<ne;++i) {
        const sdb::Coord2 & k = egs[i].vi;
        
        glVertex3fv((const float *)&ns[k.x].pos);
        glVertex3fv((const float *)&ns[k.y].pos);
    }
    glEnd();
    
}

template<typename Tn, typename Te>
void DrawGraph<Tn, Te>::drawNode(const AGraph<Tn, Te > * gph)
{
    const int & nn = gph->numNodes();
    const Tn * ns = gph->nodes();
    
    Float4 row[4];
    row[0].set(m_nodeSize,0,0,0);
    row[1].set(0,m_nodeSize,0,0);
    row[2].set(0,0,m_nodeSize,0);
    row[3].set(1,1,0,0);
    
    m_instancer->programBegin();

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
        
	for(int i=0;i<nn;++i) {
    
        const Vector3F & p = ns[i].pos;
        row[0].w = p.x;
        row[1].w = p.y;
        row[2].w = p.z;
    
	    glMultiTexCoord4fv(GL_TEXTURE1, (const float *)&row[0]);
	    glMultiTexCoord4fv(GL_TEXTURE2, (const float *)&row[1]);
	    glMultiTexCoord4fv(GL_TEXTURE3, (const float *)&row[2]);
        
        getNodeColor((float *)&row[3], ns[i].val);
		m_instancer->setDiffueColorVec((const float *)&row[3]);
	    
	    drawAGlyph();
        
	}
	
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
	
	m_instancer->programEnd();
    
}

template<typename Tn, typename Te>
void DrawGraph<Tn, Te>::getNodeColor(float * col, const float & val)
{
    if(val > 1e7f) {
        col[0] = col[1] = col[2] = .1f;
    } else if (val < 0.f) {
        float a = -val / 16.f;
        if(a > 1.f) {
            a = 1.f;
        }
        col[0] = 0.f;
        col[1] = a;
        col[2] = 1.f;
    } else {
        float a = val / 16.f;
        if(a > 1.f) {
            a = 1.f;
        }
        col[0] = 1.f;
        col[1] = a;
        col[2] = 0.f;
    }
}

}
#endif
