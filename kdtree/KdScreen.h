#pragma once

#include <ViewFrame.h>

template<typename T>
class KdScreen {
	ViewFrame m_subFrames[1<<12];
	ViewFrame m_base;
	unsigned char * m_rgba;
	float * m_z;
	
public:
    KdScreen();
	virtual ~KdScreen();
	
	void create(int w, int h);
	void setView(const Frustum & f);
    void getVisibleFrames(T * tree);
    
private:
    void clear();
};

template<typename T>
KdScreen<T>::KdScreen() 
{
    m_rgba = NULL;
    m_z = NULL;
}

template<typename T>
KdScreen<T>::~KdScreen() 
{
    clear();
}

template<typename T>
void KdScreen<T>::clear()
{
    if(m_rgba) delete[] m_rgba;
    if(m_z) delete[] m_z;
}

template<typename T>
void KdScreen<T>::create(int w, int h)
{
    clear();
    m_rgba = new unsigned char[w * h * 4];
    m_z = new float[w * h];
    
    m_base.setRect(0, 0, w-1, h-1);
}

template<typename T>
void KdScreen<T>::setView(const Frustum & f)
{
    m_base.setView(f);
}

template<typename T>
void KdScreen<T>::getVisibleFrames(T * tree)
{
    
}
