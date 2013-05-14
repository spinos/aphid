#pragma once
class Vector3F;
class BaseQuad {
public:
    BaseQuad();
    virtual ~BaseQuad();
    void setCorner(Vector3F p, int i);
	Vector3F getCorner(int i) const;
	
	Vector3F interpolate(float u, float v) const;

    Vector3F * _corners;
private:
};
