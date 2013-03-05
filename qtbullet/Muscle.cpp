#include "Muscle.h"

Muscle::Muscle() {}
Muscle::~Muscle()
{
    m_fascicles.clear();
}

void Muscle::addFacicle(const MuscleFascicle &fascicle)
{
    m_fascicles.push_back(fascicle);
}

void Muscle::create(	btSoftBodyWorldInfo& worldInfo)
{
    int nodeCount = numVertices();
    
    btVector3*		x=new btVector3[nodeCount];
	btScalar*		m=new btScalar[nodeCount];
	
	int nFascicle = 0;
	for(std::vector<MuscleFascicle>::iterator it = m_fascicles.begin() ; it != m_fascicles.end(); ++it) {
        for(int i = 0; i < (*it).numVertices(); i++) {
            int vert = vertexIdx(nFascicle, i);
            x[vert] = (*it).vertexAt(i);
            m[vert] = 0.1f;
        }
        nFascicle++;
    }
    
	m_dynBody = new btSoftBody(&worldInfo,nodeCount,x,m);
	
	delete[] x;
	delete[] m;
	
	nFascicle = 0;
	for(std::vector<MuscleFascicle>::iterator it = m_fascicles.begin() ; it != m_fascicles.end(); ++it) {
        for(int i = 1; i < (*it).numVertices(); i++) {
            int vert = vertexIdx(nFascicle, i);
            m_dynBody->appendLink(vert-1, vert);
        }
        nFascicle++;
    }
}

int Muscle::numVertices() const
{
    int sum = 0;
    for(std::vector<MuscleFascicle>::const_iterator it = m_fascicles.begin() ; it != m_fascicles.end(); ++it) {
        sum += (*it).numVertices();
    }
    return sum;
}

void Muscle::addAnchor(btRigidBody* target, int fascicle, int end)
{
    int idx;
    if(end == 0) idx = fascicleStart(fascicle); 
    else idx = fascicleEnd(fascicle); 
    
    if(target) m_dynBody->appendAnchor(idx, target);
    else m_dynBody->setMass(idx, 0);
}

btSoftBody* Muscle::getSoftBody()
{
    return m_dynBody;
}

int Muscle::vertexIdx(int fascicle, int end) const
{
    return fascicleStart(fascicle) + end;
}

int Muscle::fascicleStart(int fascicle) const
{    
    int nFascicle = 0;
    int res = 0;
	for(std::vector<MuscleFascicle>::const_iterator it = m_fascicles.begin() ; it != m_fascicles.end(); ++it) {
	    if(nFascicle == fascicle)
	        return res;
        res += (*it).numVertices();
        nFascicle++;
    }
    return res;
}

int Muscle::fascicleEnd(int fascicle) const
{    
    return fascicleStart(fascicle) + m_fascicles.at(fascicle).numVertices() - 1;
}
