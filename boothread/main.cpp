/*
 *  main.cpp
 *  
 *
 *  Created by jian zhang on 10/25/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */

#include <iostream>  
#include <boost/thread.hpp>   
#include <boost/date_time.hpp>  
#include <boost/timer.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
int *sharedNum; 
int sharedLen = 65536*128; 

typedef boost::shared_mutex Lock;
typedef boost::unique_lock< boost::shared_mutex > WrtieLock;
typedef boost::shared_lock< boost::shared_mutex >  ReadLock;

Lock myLock; 

class ThreadClass {
    public:
        ThreadClass(); // Constructor
        ~ThreadClass(); // Destructor
        void Run(int start, int c);
        void scan(int c);
		int sum;
		static int *Src;
	private:
};

int *ThreadClass::Src = 0;

ThreadClass::ThreadClass() { 
} // Constructor
ThreadClass::~ThreadClass() { } // Destructor

void ThreadClass::scan(int c) {
     //std::cout << "Worker: running" << std::endl; 


/*
	int *r = start;
	int *b = new int[c];
	for(int j=0; j < c; j++) {
		
		b[j] = *r;
		
		r++;
	}*/
	
	for(int j=0; j < 2000; j++) {
	sum = 0;
		for(int i=0; i < c; i++) {
		sum += Src[i];
		}
	}
	//delete[] b;
	//std::cout << "sum up " << sum << std::endl;
    // Pretend to do something useful... 
    //boost::this_thread::sleep(workTime);          
    //std::cout << "Worker: finished" << std::endl; 
}

void ThreadClass::Run(int start, int c) {
     //std::cout << "Worker: running" << std::endl; 

	int *r = Src + start;
	//int *b = new int[c];
	/*myLock.lock();
	for(int j=0; j < c; j++) {
		
		b[j] = *r;
		
		r++;
	}
	myLock.unlock();*/
	
	for(int j=0; j < 400; j++) {
	sum = 0;
		for(int i=0; i < c; i++) {
		sum += r[i];
		}
	}
	//delete[] b;
	//std::cout << "sum up " << sum << std::endl;
    // Pretend to do something useful... 
    //boost::this_thread::sleep(workTime);          
    //std::cout << "Worker: finished" << std::endl; 
}
      
int main(int argc, char* argv[])  
{  	
	sharedNum = new int[sharedLen];
	for(int i = 0; i < sharedLen; i++) {
		sharedNum[i] = 1;
	}
	
	ThreadClass::Src = sharedNum;
	
	boost::timer met;
	std::cout<<"sequence scan start\n ";
	met.restart();
	ThreadClass t0;
	t0.scan(sharedLen);
	std::cout<<"sequence sum "<<t0.sum<<std::endl;
	std::cout<<"took "<<met.elapsed() * 1000<<"ms\n";
    std::cout << "main: startup" << std::endl; 
	
	met.restart();
	
	const int nt = 64;
	ThreadClass t[nt];
	boost::thread tr[nt];
	
	for(int ti = 0; ti < nt; ti++) {
		tr[ti] = boost::thread(boost::bind(&ThreadClass::Run, &t[ti], sharedLen/nt*ti, sharedLen/nt));
	}
	std::cout << "main: waiting for thread" << std::endl;          
	for(int ti = 0; ti < nt; ti++) {
		tr[ti].join();
	}
      
    std::cout << "main: done" << std::endl; 
	std::cout<<"took "<<met.elapsed() * 1000<<"ms\n";
	int sum = 0;
	for(int ti = 0; ti < nt; ti++) {
		sum += t[ti].sum;
	}
	std::cout<<"thread sum "<<sum<<std::endl;
    return 0;  
}