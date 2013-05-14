#include <QtGui>
#include <QtOpenGL>

#include <math.h>

#include "KnitView.h"
#include "KnitPatch.h"
#include <EasemodelUtil.h>

//! [0]
KnitView::KnitView(QWidget *parent)
    : Base3DView(parent)
{	
	_model = new PatchMesh;
#ifdef WIN32
    ESMUtil::ImportPatch("D:/aphid/mdl/sweater.m", _model);
#else
	ESMUtil::ImportPatch("/Users/jianzhang/aphid/catmullclark/plane.m", _model);
#endif

	Vector3F* cvs = _model->getVertices();
	Vector3F* normal = _model->getNormals();
	
	unsigned* valence = _model->vertexValence();
	unsigned* patchV = _model->patchVertices();
	char* patchB = _model->patchBoundaries();
	float* ucoord = _model->us();
	float* vcoord = _model->vs();
	unsigned * uvIds = _model->uvIds();
	const int numFace = _model->numPatches();
	
	m_knit = new KnitPatch[numFace];
	
	unsigned acc = 0;
	for(int i = 0; i < numFace; i++) {
	    float u[4];
	    for(int j = 0; j < 4; j++) {
	        unsigned uv = uvIds[i * 4 + j];
	        u[j] = ucoord[uv];
	    }
	    
	    unsigned lou = biggestDu(u);
	    
	    for(int j = 0; j < 4; j++) {
	        unsigned lj = lou + j;
	        lj = lj%4;
	        unsigned v = _model->m_quadIndices[acc + lj];
	        m_knit[i].setCorner(cvs[v], j);
	        
	    }
	    acc += 4;
	    m_knit[i].createYarn();
	}
}
//! [0]

//! [1]
KnitView::~KnitView()
{
}

void KnitView::clientDraw()
{
	//getDrawer()->edge(_model);
	getDrawer()->setWired(1);
	const int numFace = _model->numPatches();	
	for(int i = 0; i < numFace; i++) {
	    glBegin(GL_LINE_STRIP);
	    for(int j = 0; j < m_knit[i].numYarnPoints(); j++) {
	        Vector3F v = m_knit[i].yarn()[j];
	        glVertex3f(v.x, v.y, v.z);
	    }
	    glEnd();
	}
}

unsigned KnitView::biggestDu(float *u) const
{
    unsigned r = 0;
    float maxdu = u[1] - u[0];
    for(unsigned i = 1; i < 4; i++) {
        unsigned i1 = i + 1;
        if(i1 > 3) i1 = 0;
        float du = u[i1] - u[i];
        if(du > maxdu) {
            maxdu = du;
            r = i;
        }
    }
    return r;
}
