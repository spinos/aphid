#include "Skin.h"
#include "modelIn.h"
Skin::Skin() {}
Skin::~Skin() {}
    
void Skin::create(btSoftBodyWorldInfo& worldInfo, const char *filename)
{
    EasyModel* model = new EasyModel(filename);
    unsigned numVertices = model->getNumVertex();
    
    btVector3*		x=new btVector3[numVertices];
	btScalar*		m=new btScalar[numVertices];
	
	float* cvs = model->getVertexPosition();
	
	unsigned i;
	for(i = 0; i < numVertices; i++) {
        x[i] = btVector3(cvs[i * 3], cvs[i * 3 + 1], cvs[i * 3 + 2]);
        m[i] = 0.1f;
    }
    
    m_dynBody = new btSoftBody(&worldInfo, numVertices,x,m);
	
	delete[] x;
	delete[] m;
	
    unsigned numFaces = model->getNumFace();
    int *faceCount = model->getFaceCount();
    int *faceConnection = model->getFaceConnection();
    
    int j;
    int acc = 0;
    for(i = 0; i < numFaces; i++) {
        for(j = 0; j <faceCount[i]; j++) {
            if(j == 0)
                m_dynBody->appendLink(faceConnection[j + faceCount[i] - 1 + acc], faceConnection[j + acc]);
            else
                m_dynBody->appendLink(faceConnection[j + acc], faceConnection[j -1 + acc]);
        }
        
        if(faceCount[i] > 3) {
            m_dynBody->appendLink(faceConnection[acc], faceConnection[2 + acc]);
            m_dynBody->appendLink(faceConnection[1 + acc], faceConnection[3 + acc]);
        }
        acc += faceCount[i];
    }
    
    delete model;
    
    m_dynBody->m_materials[0]->m_kLST	= 0.49;
    m_dynBody->m_cfg.kDP			=	0.2;
    //m_dynBody->m_materials[0]->m_kAST	= 0.9;
    //m_dynBody->m_materials[0]->m_kVST	= 0.999326945;
    
    //m_dynBody->generateBendingConstraints(2, m_dynBody->m_materials[0]); 
}

void Skin::drag(float x, float y, float z)
{
    m_dynBody->m_nodes[67].m_x += btVector3(x, y, z);
}

void Skin::addAnchor(btRigidBody* target, int idx)
{
    if(target) m_dynBody->appendAnchor(idx, target);
    else m_dynBody->setMass(idx, 0);
}

btSoftBody* Skin::getSoftBody()
{
    return m_dynBody;
}
