#ifndef TTG_PARAMETER_H
#define TTG_PARAMETER_H

#include <string>

namespace ttg {

class Parameter {
	
	std::string m_inFileName;
	std::string m_outFileName;
	
public:
	enum Operation {
		kHelp = 0,
		kKdistance = 1
	};
	
	Parameter(int argc, char *argv[]);
	virtual ~Parameter();
	
	Operation operation() const;
	const std::string & inFileName() const;
	const std::string & outFileName() const;
	static void PrintHelp();
	
protected:

private:
	Operation m_operation;
};

}
#endif        //  #ifndef LFPARAMETER_H

