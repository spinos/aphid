#ifndef TTI_PARAMETER_H
#define TTI_PARAMETER_H

#include <string>
#include <vector>

namespace tti {

class Parameter {
	
	std::string m_inFileName;
	
public:
	enum Operation {
		kHelp = 0,
		kTriangulateAsset = 1
	};
	
	Parameter(int argc, char *argv[]);
	virtual ~Parameter();
	
	Operation operation() const;
	const std::string & inFileName() const;
	static void PrintHelp();
	
protected:

private:
	Operation m_operation;
};

}
#endif        //  #ifndef LFPARAMETER_H

