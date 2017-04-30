#ifndef BASESAMPLER_H
#define BASESAMPLER_H
#include <iostream>
class BaseSampler {
public:
	BaseSampler() {}
    
    virtual void set(unsigned * vs, float * cs) {}
    
	template<typename T>	
    T evaluate(T * data) const
	{ T a; return a; }
};
	
class LineSampler : public BaseSampler {
public:
    LineSampler() {}
    
    virtual void set(unsigned * vs, float * cs) {}
    
    template<typename T>
    T evaluate(T * data) const 
    {
        return data[m_vertices[0]] * m_contributes[0]
                + data[m_vertices[1]] * m_contributes[1];
    }
    
private:
    unsigned m_vertices[2];
    float m_contributes[2];
};
	
class TriangleSampler : public BaseSampler {
public:
    TriangleSampler() {}
    
    virtual void set(unsigned * vs, float * cs) {}
    
    template<typename T>
    T evaluate(T * data) const 
    {
        return data[m_vertices[0]] * m_contributes[0]
                + data[m_vertices[1]] * m_contributes[1]
                + data[m_vertices[2]] * m_contributes[2];
    }
    
private:
    unsigned m_vertices[3];
    float m_contributes[3];
};
	
class TetrahedronSampler : public BaseSampler {
public:
    TetrahedronSampler() {}
    
    virtual void set(unsigned * vs, float * cs) 
    {
        m_vertices[0] = vs[0];
        m_vertices[1] = vs[1];
        m_vertices[2] = vs[2];
        m_vertices[3] = vs[3];
        m_contributes[0] = cs[0];
        m_contributes[1] = cs[1];
        m_contributes[2] = cs[2];
        m_contributes[3] = cs[3];
/*
        std::cout<<" "<<m_vertices[0]
        <<","<<m_vertices[1]
        <<","<<m_vertices[2]
        <<","<<m_vertices[3]
        <<" "<<m_contributes[0]
        <<","<<m_contributes[1]
        <<","<<m_contributes[2]
        <<","<<m_contributes[3];*/
    }
    
    template<typename T>
    T evaluate(T * data) const 
    {
        return data[m_vertices[0]] * m_contributes[0]
                + data[m_vertices[1]] * m_contributes[1]
                + data[m_vertices[2]] * m_contributes[2]
                + data[m_vertices[3]] * m_contributes[3];
    }
    
private:
    unsigned m_vertices[4];
    float m_contributes[4];
};
#endif        //  #ifndef BASESAMPLER_H

