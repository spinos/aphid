#ifndef APH_SAMPLE_TIME_H
#define APH_SAMPLE_TIME_H

#include <sdb/Sequence.h>
#include <fstream>
#include <string>
#include <boost/lexical_cast.hpp>

namespace aphid {

template<typename ChildType>
class ASampleTime : public sdb::Sequence<int>
{
    typedef sdb::Sequence<int> ParentTyp;
public:
    ASampleTime(sdb::Entity * parent = NULL);
    virtual ~ASampleTime();
    
    bool loadFromFile(const char * fileName);
    
    ChildType * value();
    const int & key() const;
    
    void insertSample(const float & smp);
    void insertSample(const int& frame, 
                    const int& subframe);
    int sampleCount();
    
    ChildType * findFrame(const int& i);
    
protected:

private:
    void insertFromStr(const std::string & stime);
    
};

template<typename ChildType>
ASampleTime<ChildType>::ASampleTime(sdb::Entity * parent) :
sdb::Sequence<int>(parent)
{}

template<typename ChildType>
ASampleTime<ChildType>::~ASampleTime()
{}

template<typename ChildType>
bool ASampleTime<ChildType>::loadFromFile(const char * fileName)
{
    std::ifstream fsm;
    fsm.open(fileName);
    if(!fsm ) {
        std::cout<<"\n ASampleTime cannot open file "<<fileName;
        return false;
    }
    std::string sline;
    while(!fsm.eof() ) {
        std::getline(fsm, sline);
        std::cout<<"\n read sample time "<<sline;
        if(sline.size() > 2) {
                insertFromStr(sline);
        }
    }
    fsm.close();
    return true;
}

template<typename ChildType>
ChildType * ASampleTime<ChildType>::value() 
{ 
	ChildType * r = dynamic_cast<ChildType *>(ParentTyp::currentIndex() );
	if(r == NULL) throw " ASampleTime value null ";
	return r;
}

template<typename ChildType>
const int & ASampleTime<ChildType>::key() const 
{ return ParentTyp::currentKey(); }

template<typename ChildType>
void ASampleTime<ChildType>::insertSample(const float & smp)
{
    int frame = smp;
    float frag = smp - frame;
    sdb::Pair<int, sdb::Entity> * p = ParentTyp::insert(frame);
	if(!p) {
	     throw " ASampleTime cannot insert";
	}
	if(!p->index) {
	    p->index = new ChildType(this);
	}
	ChildType *d = static_cast<ChildType *>(p->index);
	try {
	    d->insert(frag);
	} catch (const char * ex) {
		std::cerr<<" ASampleTime insert caught "<<ex;
	} catch(...) {
		std::cerr<<" ASampleTime insert caught something";
	}
}

template<typename ChildType>
int ASampleTime<ChildType>::sampleCount()
{
	int c = 0;
	begin();
	while(!end() ) {
	
		c += value()->size();
		next();
	}
	return c;
}

template<typename ChildType>
void ASampleTime<ChildType>::insertFromStr(const std::string & stime)
{
    try {
        float ftime = boost::lexical_cast<float>(stime);
        insertSample(ftime);
        
    } catch (boost::bad_lexical_cast &) {
            std::cout<<"\n bad cast "<<stime;
        }
}

template<typename ChildType>
ChildType * ASampleTime<ChildType>::findFrame(const int& i)
{
    sdb::Pair<sdb::Entity *, sdb::Entity> p = ParentTyp::findEntity(i);
	if(p.index) {
		ChildType * g = static_cast<ChildType *>(p.index);
		return g;
	}
	return NULL;
}

template<typename ChildType>
void ASampleTime<ChildType>::insertSample(const int& frame, 
                    const int& subframe)
{
    sdb::Pair<int, sdb::Entity> * p = ParentTyp::insert(frame);
	if(!p) {
	     throw " ASampleTime cannot insert";
	}
	if(!p->index) {
	    p->index = new ChildType(this);
	}
	ChildType *d = static_cast<ChildType *>(p->index);
	try {
	    d->insert(subframe);
	} catch (const char * ex) {
		std::cerr<<" ASampleTime insert caught "<<ex;
	} catch(...) {
		std::cerr<<" ASampleTime insert caught something";
	}
}

typedef sdb::Sequence<float> SubframeSampleTimeF;
typedef ASampleTime<SubframeSampleTimeF > ASampleTimeF;
typedef sdb::Sequence<int> SubframeSampleTimeI;
typedef ASampleTime<SubframeSampleTimeI > ASampleTimeI;

}

#endif
