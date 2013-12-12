#pragma once

#include <AllMath.h>

class BaseSphere {
public:
    BaseSphere();
    virtual ~BaseSphere();
    
    void setCenter(const Vector3F & pos);
    void setRadius(float r);
    
    Vector3F center() const;
    float radius() const;
    
    char inside(const BaseSphere & another, float & delta) const;
    char expand(const BaseSphere & another);
private:
    Vector3F m_center;
    float m_radius;
};
