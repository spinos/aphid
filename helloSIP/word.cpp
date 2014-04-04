#include "word.h"

#include <iostream>
#include <sstream>

namespace MY {
Word::Word()
{
}

std::string Word::reversed(const std::string & w) const
{
	std::string r = std::string(w);
	int count = r.size();
	for(int i = 0; i < count; i++)
		r[i] = w[count - 1  - i];	

	return r;
}

std::string Word::str(const std::string & name) const
{
	std::string res("hello ");
	res = res + name;
	return res;
}

std::string Word::strvec(const std::vector<std::string> & name) const
{
	std::string res("hello ");
	std::vector<std::string>::const_iterator it = name.begin();
	for(; it != name.end(); ++it) {
		res = res + " and ";
		res = res + *it;
	}
	return res;
}

struct ooops : std::exception {
  const char* what() const throw() {return "Ooops!\n";}
};

int Word::someerr() const
{
    throw ooops();
}

};
