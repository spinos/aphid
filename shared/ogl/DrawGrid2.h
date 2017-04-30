#ifndef APH_DRAWGRID2_H
#define APH_DRAWGRID2_H

#include <boost/scoped_array.hpp>

namespace aphid {

class BoundingBox;
class DrawGrid2 {
    
    boost::scoped_array<float> m_vertexNormals;
	boost::scoped_array<float> m_vertexPoints;
	boost::scoped_array<float> m_vertexColors;
	int m_numVertices;
	
public:
    DrawGrid2();
    virtual ~DrawGrid2();
    
/// for each level cell
/// buffer face has no neighbor
    template<typename T>
    void create(T * grd, const int & level);
    void setUniformColor(const float * col);
    
    void drawSolidGrid() const;
    
protected:
    void setBoxFace(float * pos,
                    float * mnl,
                    const BoundingBox & bx,
                    const int & iface);
    
private:
};

template<typename T>
void DrawGrid2::create(T * grd, const int & level)
{
    int nfaces = 0;
    grd->begin();
    while(!grd->end() ) {
        if(grd->key().w == level) {
	
			for(int i=0;i<6;++i) {
			    if(!grd->value()->neighbor(i)) {
			        nfaces++;
			    }
			}
        }
        
        if(grd->key().w > level) {
			break;
		}
			
		grd->next();
	}
	std::cout<<"\n DrawGrid2::create nfaces "<<nfaces;
/// 2 triangles * 3 vertices * 3 floats
	m_vertexPoints.reset(new float[nfaces * 18]);
	m_vertexNormals.reset(new float[nfaces * 18]);
	m_vertexColors.reset(new float[nfaces * 18]);
	
	BoundingBox cbx;
	nfaces = 0;
    grd->begin();
    while(!grd->end() ) {
        sdb::Coord4 c = grd->key();
        if(c.w == level) {
	
            grd->getCellBBox(cbx, c);
            
			for(int i=0;i<6;++i) {
			    if(!grd->value()->neighbor(i)) {
			        float * pr = &m_vertexPoints[nfaces * 18];
			        float * nr = &m_vertexNormals[nfaces * 18];
			        
			        setBoxFace(pr, nr, cbx, i);
			        nfaces++;
			    }
			}
        }
        
        if(c.w > level) {
			break;
		}
			
		grd->next();
	}
	m_numVertices = nfaces * 6;
}

}
#endif        //  #ifndef DRAWGRID2_H

