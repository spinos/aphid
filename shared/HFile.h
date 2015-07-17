/*
 *  HFile.h
 *  mallard
 *
 *  Created by jian zhang on 10/10/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#pragma once
#include <HDocument.h>
#include <BaseFile.h>
#include <vector>

class HFile : public BaseFile {
public:
	HFile();
	HFile(const char * name);
	
	virtual bool doCreate(const std::string & fileName);
	virtual bool doRead(const std::string & fileName);
	virtual void doClose();
	
	void flush();
	
protected:
	void useDocument();
	void setDocument(const HDocument & doc);
    bool entityExists(const std::string & name);
    
    template<typename Tb, typename Th>
    static bool LsNames2(std::vector<std::string> & dst, Tb * parent)
    {
        std::vector<std::string > aNames;
        parent->lsTypedChild<Tb>(aNames);
        std::vector<std::string>::const_iterator ita = aNames.begin();
        
        for(;ita!=aNames.end();++ita) {
            Tb child(*ita);
            LsNames2<Tb, Th>(dst, &child);
            child.close();
        }
        
        std::vector<std::string > hNames;
        parent->lsTypedChild<Th>(hNames);
        std::vector<std::string>::const_iterator itb = hNames.begin();
        
        for(;itb!=hNames.end();++itb)
            dst.push_back(*itb);
        
        return true;   
    }
private:
	HDocument m_doc;
};