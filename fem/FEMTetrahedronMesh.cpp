#include "FEMTetrahedronMesh.h"

FEMTetrahedronMesh::FEMTetrahedronMesh() : m_density(1000.f) {}
FEMTetrahedronMesh::~FEMTetrahedronMesh() {}

void FEMTetrahedronMesh::setDensity(float x) { m_density = x; }
void FEMTetrahedronMesh::generateBlocks(unsigned xdim, unsigned ydim, unsigned zdim, float width, float height, float depth)
{
    m_totalPoints = (xdim+1)*(ydim+1)*(zdim+1);
	m_X = new Vector3F[m_totalPoints];
	m_Xi= new Vector3F[m_totalPoints];
	m_mass = new float[m_totalPoints];
	
	int ind=0;
	float hzdim = zdim/2.0f;
	for(unsigned x = 0; x <= xdim; ++x) {
		for (unsigned y = 0; y <= ydim; ++y) {
			for (unsigned z = 0; z <= zdim; ++z) {			  
				m_X[ind] = Vector3F(width*x, height*z, depth*y);            
				m_Xi[ind] = m_X[ind];
				ind++;
			}
		}
	}
	//offset the m_tetrahedronl mesh by 0.5 units on y axis
	//and 0.5 of the depth in z axis
	for(unsigned i=0;i< m_totalPoints;i++) {
		m_X[i].y += 2.5;		
		m_X[i].z -= hzdim*depth; 
	}
	
	m_totalTetrahedrons = 5 * xdim * ydim * zdim;
	
	m_tetrahedron = new Tetrahedron[m_totalTetrahedrons];
	Tetrahedron * t = &m_tetrahedron[0];
	for (unsigned i = 0; i < xdim; ++i) {
		for (unsigned j = 0; j < ydim; ++j) {
			for (unsigned k = 0; k < zdim; ++k) {
				unsigned p0 = (i * (ydim + 1) + j) * (zdim + 1) + k;
				unsigned p1 = p0 + 1;
				unsigned p3 = ((i + 1) * (ydim + 1) + j) * (zdim + 1) + k;
				unsigned p2 = p3 + 1;
				unsigned p7 = ((i + 1) * (ydim + 1) + (j + 1)) * (zdim + 1) + k;
				unsigned p6 = p7 + 1;
				unsigned p4 = (i * (ydim + 1) + (j + 1)) * (zdim + 1) + k;
				unsigned p5 = p4 + 1;
				// Ensure that neighboring tetras are sharing faces
				if ((i + j + k) % 2 == 1) {
					addTetrahedron(t++, p1,p2,p6,p3);
					addTetrahedron(t++, p3,p6,p4,p7);
					addTetrahedron(t++, p1,p4,p6,p5);
					addTetrahedron(t++, p1,p3,p4,p0);
					addTetrahedron(t++, p1,p6,p4,p3); 
				} else {
					addTetrahedron(t++, p2,p0,p5,p1);
					addTetrahedron(t++, p2,p7,p0,p3);
					addTetrahedron(t++, p2,p5,p7,p6);
					addTetrahedron(t++, p0,p7,p5,p4);
					addTetrahedron(t++, p2,p0,p7,p5); 
				}
			}
		}
	}
}

void FEMTetrahedronMesh::addTetrahedron(Tetrahedron *t, unsigned i0, unsigned i1, unsigned i2, unsigned i3) 
{
	t->indices[0]=i0;
	t->indices[1]=i1;
	t->indices[2]=i2;
	t->indices[3]=i3; 
}

unsigned FEMTetrahedronMesh::numTetrahedra() const { return m_totalTetrahedrons; }
unsigned FEMTetrahedronMesh::numPoints() const { return m_totalPoints; }
FEMTetrahedronMesh::Tetrahedron * FEMTetrahedronMesh::tetrahedra() {return m_tetrahedron; } 
Vector3F * FEMTetrahedronMesh::X() { return m_X; }
Vector3F * FEMTetrahedronMesh::Xi() { return m_Xi; }
float FEMTetrahedronMesh::getTetraVolume(Vector3F e1, Vector3F e2, Vector3F e3) {
	return  e1.dot( e2.cross( e3 ) )/ 6.0f;
}
float * FEMTetrahedronMesh::M() { return m_mass; }
float FEMTetrahedronMesh::getTetraVolume(Tetrahedron & tet) const
{
    Vector3F x0 = m_X[tet.indices[0]];
    Vector3F x1 = m_X[tet.indices[1]];
    Vector3F x2 = m_X[tet.indices[2]];
    Vector3F x3 = m_X[tet.indices[3]];
    Vector3F e10 = x1-x0;
	Vector3F e20 = x2-x0;
	Vector3F e30 = x3-x0;
	return getTetraVolume(e10,e20,e30);
}

float FEMTetrahedronMesh::volume() const
{
    float r = 0.f;
    for(unsigned i = 0; i < m_totalTetrahedrons; i++)
        r += getTetraVolume(m_tetrahedron[i]);

    return r;
}

float FEMTetrahedronMesh::volume0() const
{
    float r = 0.f;
    for(unsigned i = 0; i < m_totalTetrahedrons; i++)
        r += m_tetrahedron[i].volume;

    return r;
}

void FEMTetrahedronMesh::recalcMassMatrix(bool * isFixed) 
{
	//This is a lumped mass matrix
	//Based on Eq. 10.106 and pseudocode in Fig. 10.9 on page 358
	for(unsigned i=0;i<m_totalPoints;i++) {
		if(isFixed[i])
			m_mass[i] = 10e10;
		else
			m_mass[i] = 1.0f/m_totalPoints;
	}
	
	unsigned a, b, c, d;
	for(int i=0;i<m_totalTetrahedrons;i++) {
		float m = (m_density * m_tetrahedron[i].volume) * 0.25f;
		Tetrahedron tet = m_tetrahedron[i];
		a = tet.indices[0];
		b = tet.indices[1];
		c = tet.indices[2];
		d = tet.indices[3];
		m_mass[a] += m;
		m_mass[b] += m;
		m_mass[c] += m;
		m_mass[d] += m;
	}	 
}

float FEMTetrahedronMesh::mass() const
{
    float r = 0.f;
    for(unsigned i=0;i<m_totalPoints;i++)
        if(m_mass[i] < 10e8) r += m_mass[i];
    return r;
}
