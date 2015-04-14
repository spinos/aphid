#include <AllMath.h>

class FEMTetrahedronMesh
{
public:
    struct Tetrahedron {
        unsigned indices[4];			//indices
        float volume;			//volume 
        float plastic[6];		//plasticity values
        Matrix33F Re;			//Rotational warp of tetrahedron.
        Matrix33F Ke[4][4];		//Stiffness element matrix
       // Vector3F e1, e2, e3;	//edges
        Vector3F B[4];			//Jacobian of shapefunctions; B=SN =[d/dx  0     0 ][wn 0  0]
                                //                                  [0    d/dy   0 ][0 wn  0]
                                //									[0     0   d/dz][0  0 wn]
                                //									[d/dy d/dx   0 ]
                                //									[d/dz  0   d/dx]
                                //									[0    d/dz d/dy]
    };
    FEMTetrahedronMesh();
    virtual ~FEMTetrahedronMesh();
    
    void setDensity(float x);
    void generateFromFile();
    void generateBlocks(unsigned xdim, unsigned ydim, unsigned zdim, float width, float height, float depth);
    void recalcMassMatrix(bool * isFixed);
    unsigned numTetrahedra() const;
    unsigned numPoints() const;
    Tetrahedron * tetrahedra();
    Vector3F * X();
    Vector3F * Xi();
    float * M();
    float volume() const;
    float volume0() const;
    float mass() const;
    
    static float getTetraVolume(Vector3F e1, Vector3F e2, Vector3F e3);
    float getTetraVolume(Tetrahedron & tet) const;
    
private:
    void addTetrahedron(Tetrahedron *t, unsigned i0, unsigned i1, unsigned i2, unsigned i3);
 
private:
    Vector3F * m_X;
	Vector3F * m_Xi;
	float * m_mass;
	Tetrahedron * m_tetrahedron;
	unsigned m_totalPoints, m_totalTetrahedrons;
	float m_density;
};

