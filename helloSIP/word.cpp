#include "word.h"

#include <iostream>
#include <sstream>

using namespace std;
namespace MY {
Word::Word()
{
}

char *Word::reverse(const char *w) const
{
	string r = string(w);
	int count = r.size();
	char *res = new char[count];
	for(int i = 0; i < count; i++) {
		res[i] = w[count - 1  - i];	
	}
	return res;
}
};
