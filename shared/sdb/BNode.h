/*
 *  BNode.h
 *  
 *
 *  Created by jian zhang on 4/24/14.
 *  Copyright 2014 __MyCompanyName__. All rights reserved.
 *
 */
#pragma once
#include <vector>
#include <map>
#include <sdb/KeyNData.h>
#include <sstream>

namespace aphid {

namespace sdb {

#define DBG_SPLIT 0

class TreeNode : public Entity
{
/// to left child 
/// or to sibling for leaf
	Entity * m_link;
	bool m_isLeaf;
	
public:
	TreeNode(Entity * parent = NULL);
	virtual ~TreeNode();
	
	bool isRoot() const;
	bool hasChildren() const;
	bool isLeaf() const;
	
	Entity * sibling() const;
	Entity * leftChild() const;
	
	void setLeaf();
	void connectSibling(Entity * another);
	void connectLeftChild(Entity * another);
	
protected:
	struct NodeIndirection {
		void reset() {
			_p = NULL;
		}
		
		void take(Entity * src) {
			_p = src;
		}
		
		Entity * give() {
			return _p;
		}
		
		Entity * _p;
	};
	
private:
	
};

template <typename KeyType, int MaxNKey = 128>
class BNode : public TreeNode, public KeyNData <KeyType, MaxNKey>
{
public:
	BNode(Entity * parent = NULL);
	virtual ~BNode();
	
	Pair<KeyType, Entity> * insert(const KeyType & x);
    Pair<Entity *, Entity> find(const KeyType & x);
	void remove(const KeyType & x);
	
    void getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const;
	BNode * firstLeaf();
	BNode * nextLeaf();
	
	Pair<Entity *, Entity> findInNode(const KeyType & x, SearchResult * result);
	
	friend std::ostream& operator<<(std::ostream &output, const BNode & p) {
        output << p.str();
        return output;
    }
	
	void dbgFind(const KeyType & x);
	bool dbgLinks(bool silient=true);
	void dbgDown() const;
	
private:	
	
	bool dataLeftTo(const KeyType & x, Pair<KeyType, Entity> & dst) const;
	
	BNode * parentNode() const;
	BNode * siblingNode() const;
	BNode * ancestor(const KeyType & x, bool & found) const;
	BNode * leftTo(const KeyType & x) const;
	BNode * rightTo(const KeyType & x, Pair<KeyType, Entity> & k) const;
	BNode * leafLeftTo(const KeyType & x);
    BNode * nextIndex(KeyType x) const;
	BNode * leftInteriorNeighbor() const;
	BNode * rightInteriorNeighbor(int & indParent) const;
	BNode * linkedNode(int idx) const;
	BNode * leftChildNode() const; 
	
	Pair<KeyType, Entity> * insertRoot(const KeyType & x);
	Pair<KeyType, Entity> * insertLeaf(const KeyType & x);
	Pair<KeyType, Entity> * insertInterior(const KeyType & x);
	
	void splitRootToLeaves();
	BNode *splitLeaf(const KeyType & x);
	
	bool removeKey(const KeyType & x);
	
	void partRoot(const Pair<KeyType, Entity> & x);
	Pair<KeyType, Entity> partData(const Pair<KeyType, Entity> & x, 
								Pair<KeyType, Entity> old[], 
								BNode * lft, BNode * rgt, bool doSplitLeaf = false);
	
	void partInterior(const Pair<KeyType, Entity> & x);
	
/// for each child, set parent
	void connectToChildren();
/// insert data, split if needed
	void bounce(const Pair<KeyType, Entity> & b);

	bool leafBalance();
	bool leafBalanceRight();
	bool leafBalanceLeft();
	void sendDataRight(int num, BNode * rgt);
	void sendDataLeft(int num, BNode * lft);
	
	void removeRoot(const KeyType & x);
	void removeLeaf(const KeyType & x);
	bool removeDataLeaf(const KeyType & x);

	bool mergeLeaf();
	bool mergeLeafRight();
	bool mergeLeafLeft();
	
	void pop(const Pair<KeyType, Entity> & x);
	void popRoot(const Pair<KeyType, Entity> & x);
	void popInterior(const Pair<KeyType, Entity> & x);
	
	void mergeData(BNode * another, int start = 0);
	
	
	bool mergeInterior(const KeyType x);
	bool mergeInteriorRight(const KeyType x);
	bool mergeInteriorLeft(const KeyType x);
	
	bool shouldLeafMerge(BNode * lft, BNode * rgt) const;
	bool shouldInteriorMerge(BNode * lft, BNode * rgt) const;
	int shouldBalance(BNode * lft, BNode * rgt) const;
	
	Pair<Entity *, Entity> findRoot(const KeyType & x);
	Pair<Entity *, Entity> findLeaf(const KeyType & x);
	Pair<Entity *, Entity> findInterior(const KeyType & x);
	
	void dbgFindInRoot(const KeyType & x);
	void dbgFindInLeaf(const KeyType & x);
	void dbgFindInInterior(const KeyType & x);
		
	const std::string str() const;
	
	void dbgUp();
	void dbgRightUp(BNode<KeyType, MaxNKey> * rgt);
	int countBalance(int a, int b);
	void behead();
	KeyType getHighestKey() const;
	void takeover();
	void leafTakeover();
	void innerTakeover();
	void compressSingular();
	void leaf3WayMerge();
	void inner3WayMerge();
	bool isUnderflow() const;
    
};

template <typename KeyType, int MaxNKey>  
BNode<KeyType, MaxNKey>::BNode(Entity * parent) : TreeNode(parent)
{
	//m_data = new Pair<KeyType, Entity>[MaxNumKeysPerNode];
	//for(int i=0;i< MaxNumKeysPerNode;++i)
     //   m_data[i].index = NULL;
}

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey>::~BNode()
{
	//for(int i=0;i< numKeys();++i) {
	//	if(m_data[i].index) delete m_data[i].index;
	//}
	//delete[] m_data;
}

template <typename KeyType, int MaxNKey>  
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::nextIndex(KeyType x) const
{
	//std::cout<<"\n find "<<x<<" in "<<*this;
	
	if(KeyNData<KeyType, MaxNKey>::firstKey() > x) {
		if(!leftChild()) std::cout<<"\n error first index";
		return static_cast<BNode *>(leftChild());
	}
	int ii;
	SearchResult s = findKey(x);
	ii = s.low;
	if(s.found > -1) ii = s.found;
	else if(KeyNData<KeyType, MaxNKey>::key(s.high) < x) ii = s.high;
	
	//std::cout<<"found "<<ii<<"\n";
	return static_cast<BNode *>(KeyNData<KeyType, MaxNKey>::index(ii) );
}

template <typename KeyType, int MaxNKey> 
Pair<KeyType, Entity> * BNode<KeyType, MaxNKey>::insert(const KeyType & x)
{
	if(isRoot()) 
		return insertRoot(x);
	
	return insertInterior(x);
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::remove(const KeyType & x)
{
	if(isRoot()) {
    
		removeRoot(x);
		compressSingular();
	}
	else if(isLeaf()) {
    
		removeLeaf(x);
	}
	else {
    
		BNode * n = nextIndex(x);
		n->remove(x);
	}
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::removeRoot(const KeyType & x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		n->remove(x);
	}
	else {
		removeKeyAndData(x);
		
		connectLeftChild(NULL);
	}
}

template <typename KeyType, int MaxNKey> 
Pair<KeyType, Entity> * BNode<KeyType, MaxNKey>::insertRoot(const KeyType & x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		if(n->isLeaf() ) {
			return n->insertLeaf(x);
		}
		return n->insertInterior(x);
	}

/// a single node
	if(KeyNData<KeyType, MaxNKey>::isFull()) {	
		splitRootToLeaves();
		BNode * n = nextIndex(x);
		return n->insertLeaf(x);
	}
	
	SearchResult sr = findKey(x);
	if(sr.found < 0) {
		sr.found = insertKey(x);
		// std::cout<<"\n bnode insert root "<<x<<" into "<<*this;
	}
	
	return KeyNData<KeyType, MaxNKey>::dataR(sr.found);
}

template <typename KeyType, int MaxNKey> 
Pair<KeyType, Entity> * BNode<KeyType, MaxNKey>::insertLeaf(const KeyType & x)
{
	SearchResult sr = findKey(x);
	if(sr.found < 0) {
		sr.found = insertKey(x);
		/// std::cout<<"\n bnode insert leaf"<<x<<" into "<<*this;
	}
	
	BNode * dst = this;
	if(KeyNData<KeyType, MaxNKey>::isFull()) {
		dst = splitLeaf(x);
		sr = dst->findKey(x);
	}
	
	return dst->dataR(sr.found);
}

template <typename KeyType, int MaxNKey> 
Pair<KeyType, Entity> * BNode<KeyType, MaxNKey>::insertInterior(const KeyType & x)
{
	//std::cout<<"\n insert inner"<<x<<" into "<<*this;
	BNode * n = nextIndex(x);
	if(n->isLeaf() ) {
		//std::cout<<"\n child is leaf"<<n<<" nk "<<n->numKeys();
		return n->insertLeaf(x);
	}
	return n->insertInterior(x);
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::splitRootToLeaves()
{
	// std::cout<<"\n split root";
	BNode * one = new BNode(this);
	one->setLeaf();
	BNode * two = new BNode(this); 
	two->setLeaf();
	
	BNode * dst = one;
	int i = 0;
	for(;i < MaxNKey; ++i) {
		
		if(i == MaxNKey / 2)	
			dst = two;
		
		dst->insertData(KeyNData<KeyType, MaxNKey>::data(i) );
		KeyNData<KeyType, MaxNKey>::dataR(i)->index = NULL;
	}
	
	/// std::cout<<"\n into "<<*one<<"\n"<<*two;
	
	connectLeftChild(one);
	KeyNData<KeyType, MaxNKey>::setNumKeys(1);
	
	Pair<KeyType, Entity> hd;
	hd.key = two->firstKey();
	hd.index = two;
	KeyNData<KeyType, MaxNKey>::setData(0, hd);
	
	/// std::cout<<"\n aft "<<*this;
	
	one->connectSibling(two);
}

/// 2nd half to new rgt and connect it to old rgt
template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::splitLeaf(const KeyType & x)
{
#if DBG_SPLIT
	std::cout<<"\n split "<<str();
#endif
	Entity * oldRgt = sibling();
	BNode * two = new BNode(parent()); 
	two->setLeaf();

	for(int i= MaxNKey / 2; i < MaxNKey; ++i) {
		two->insertData(KeyNData<KeyType, MaxNKey>::data(i) );
		KeyNData<KeyType, MaxNKey>::dataR(i)->index = NULL;
	}
		
	KeyNData<KeyType, MaxNKey>::setNumKeys(MaxNKey / 2);
	
	connectSibling(two);
	if(oldRgt) two->connectSibling(oldRgt);
#if DBG_SPLIT	
	std::cout<<"\n into "<<str()<<" and "<<two->str();
#endif	
	Pair<KeyType, Entity> b;
	b.key = two->firstKey();
	b.index = two;
	parentNode()->bounce(b);
	
	if(two->hasKey(x)) return two;
	return this;
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::bounce(const Pair<KeyType, Entity> & b)
{	
	if(KeyNData<KeyType, MaxNKey>::isFull()) {
		if(isRoot()) 
			partRoot(b);
		else
			partInterior(b);
	}
	else
		insertData(b);
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::getChildren(std::map<int, std::vector<Entity *> > & dst, int level) const
{
	if(isLeaf()) return;
	if(!hasChildren()) return;
	dst[level].push_back(leftChild());
	for(int i = 0;i < KeyNData<KeyType, MaxNKey>::numKeys(); i++) {
		dst[level].push_back(KeyNData<KeyType, MaxNKey>::index(i));
    }
		
	level++;
		
	BNode * n = static_cast<BNode *>(leftChild());
	n->getChildren(dst, level);
	for(int i = 0;i < KeyNData<KeyType, MaxNKey>::numKeys(); i++) {
		n = static_cast<BNode *>(KeyNData<KeyType, MaxNKey>::index(i));
		n->getChildren(dst, level);
	}
}

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::firstLeaf()
{
	if(isRoot()) { 
		if(hasChildren())
			return static_cast<BNode *>(leftChild())->firstLeaf();
		else 
			return this;
			
	}
	
	if(isLeaf())
		return this;
	
	BNode * c = dynamic_cast<BNode *>(leftChild());
	if(c==NULL) {
		throw "BNode first leaf null";
	}
	return c->firstLeaf();
}

template <typename KeyType, int MaxNKey>
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::nextLeaf() 
{ 
	return dynamic_cast<BNode *>(sibling());
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::connectToChildren()
{
	if(!hasChildren()) return;
	Entity * n = leftChild();
	n->setParent(this);
	for(int i = 0;i < KeyNData<KeyType, MaxNKey>::numKeys(); i++) {
		n = KeyNData<KeyType, MaxNKey>::index(i);
		n->setParent(this);
	}
}

/// part into 2 interials
template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::partRoot(const Pair<KeyType, Entity> & b)
{
#if DBG_SPLIT
	//std::cout<<"\n part root "<<str()<<"\n add index "<<static_cast<BNode *>(b.index)->firstKey();
#endif

#if DBG_SPLIT
	//std::cout<<"\n right to "<<static_cast<BNode *>(m_data[numKeys() - 1].index)->firstKey();
#endif	

	BNode * one = new BNode(this);
	BNode * two = new BNode(this);
	
	BNode * dst = one;
	const int midI = MaxNKey / 2;
	int i=0;
	for(; i < MaxNKey; ++i) {
		if(i == midI)	
			dst = two;
		else 
			dst->insertData(KeyNData<KeyType, MaxNKey>::data(i) );
	}
	
	one->connectLeftChild(leftChild());
	
	if(b.key > KeyNData<KeyType, MaxNKey>::data(midI).key) {
#if DBG_SPLIT
		std::cout<<"\n b.k "<<b.key<<" > 1st.k "<<m_data[midI].key;
#endif
		two->insertData(b);
		
	}
	else {
#if DBG_SPLIT
		std::cout<<"\n b.k "<<b.key<<" < 1st.k "<<m_data[midI].key;
#endif
		one->insertData(b);
	}
	
	two->connectLeftChild(KeyNData<KeyType, MaxNKey>::data(midI).index);

#if DBG_SPLIT
	std::cout<<"\n two n k "<<two->numKeys();
	std::cout<<" <- "<<static_cast<BNode *>(two->leftChild() )->firstKey();
	for(int i=0;i<two->numKeys(); ++i) {
		std::cout<<" -> "<<static_cast<BNode *>(two->index(i) )->firstKey();
	}
#endif	
	
	connectLeftChild(one);
	
	Pair<KeyType, Entity> c;
	c.key = KeyNData<KeyType, MaxNKey>::data(midI).key;
	c.index = two;
	KeyNData<KeyType, MaxNKey>::setData(0, c);
	KeyNData<KeyType, MaxNKey>::setNumKeys(1);
	
#if DBG_SPLIT
	//std::cout<<"\n after "<<*this;
#endif
	one->connectToChildren();
#if DBG_SPLIT	
	//std::cout<<"\n connect right child";
#endif
	two->connectToChildren();
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::partInterior(const Pair<KeyType, Entity> & x)
{
#if DBG_SPLIT
	//std::cout<<"part interior "<<*this;
#endif
	BNode * rgt = new BNode(parent());
	
	Pair<KeyType, Entity> old[MaxNKey];
	for(int i=0; i < MaxNKey; i++)
		old[i] = KeyNData<KeyType, MaxNKey>::data(i);
	
	KeyNData<KeyType, MaxNKey>::setNumKeys(0);
	Pair<KeyType, Entity> p = partData(x, old, this, rgt);
	
#if DBG_SPLIT
	//std::cout<<" into "<<*this<<" and "<<*rgt;
#endif	
	connectToChildren();
	
	rgt->connectLeftChild(p.index);

	rgt->connectToChildren();
	
	Pair<KeyType, Entity> b;
	b.key = p.key;
	b.index = rgt;
	parentNode()->bounce(b);
}

template <typename KeyType, int MaxNKey> 
Pair<KeyType, Entity> BNode<KeyType, MaxNKey>::partData(const Pair<KeyType, Entity> & x, 
											Pair<KeyType, Entity> old[], 
											BNode * lft, BNode * rgt, bool doSplitLeaf)
{
	Pair<KeyType, Entity> res, q;
	BNode * dst = rgt;
	
	int numKeysRight = 0;
	bool inserted = false;
	int i = MaxNKey - 1;

	for(;i >= 0; i--) {
		if(x.key > old[i].key && !inserted) {
			q = x;
			i++;
			inserted = true;
		}
		else
			q = old[i];
			
		numKeysRight++;
		
		if(numKeysRight == MaxNKey / 2 + 1) {
			if(doSplitLeaf)
				dst->insertData(q);
				
			dst = lft;
			res = q;
		}
		else
			dst->insertData(q);
	}
	
	if(!inserted)
		dst->insertData(x);
			
	return res;
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::leafBalance()
{
	if(leafBalanceRight()) return true;
	return leafBalanceLeft();
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::leafBalanceRight()
{
    BNode * rgt = siblingNode();
	if(!rgt) return false;
	
	const Pair<KeyType, Entity> s = rgt->firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return false;
	
	int n = shouldBalance(this, rgt);
	if(n == 0) return false;
	
	// std::cout<<"\n balance right "<<n;
	
	Pair<KeyType, Entity> old = rgt->firstData();
	if(n < 0) sendDataRight(-n, rgt);
	else siblingNode()->sendDataLeft(n, this);
	
	crossed->replaceKey(old.key, rgt->firstData().key);
	
	// std::cout<<"\nbalanced right "<<*this<<*siblingNode();
	return true;
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::leafBalanceLeft()
{return false;
	const Pair<KeyType, Entity> s = KeyNData<KeyType, MaxNKey>::firstData();
	
	bool found = false;
	BNode * crossed = ancestor(s.key, found);
	
	if(!found) return false;
	
	BNode * leftSibling = crossed->leafLeftTo(s.key);
	
	if(leftSibling == this) return false;
	
	int k = shouldBalance(leftSibling, this);
	if(k == 0) return false;
	
	Pair<KeyType, Entity> old = KeyNData<KeyType, MaxNKey>::firstData();
	if(k < 0) leftSibling->sendDataRight(-k, this);
	else this->sendDataLeft(k, leftSibling);
	
	crossed->replaceKey(old.key, KeyNData<KeyType, MaxNKey>::firstData().key);
	
	// std::cout<<"\nbalanced left "<<*leftSibling<<*this;
	return true;
}

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::ancestor(const KeyType & x, bool & found) const
{
    if(isRoot()) return NULL;
	if(parentNode()->hasKey(x)) {
		found = true;
		return parentNode();
	}
	
	return parentNode()->ancestor(x, found);
}

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::leftTo(const KeyType & x) const
{
	if(KeyNData<KeyType, MaxNKey>::isSingular() ) return static_cast<BNode *>(leftChild());
	int i = keyLeft(x);
	if(i < 0) return static_cast<BNode *>(leftChild());
	return static_cast<BNode *>(KeyNData<KeyType, MaxNKey>::index(i));
}

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::rightTo(const KeyType & x, Pair<KeyType, Entity> & k) const
{
	int ii = keyRight(x);
	if(ii > -1) {
		k = KeyNData<KeyType, MaxNKey>::data(ii);
		return static_cast<BNode *>(KeyNData<KeyType, MaxNKey>::data(ii).index);
	}
	return NULL;
}

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::leafLeftTo(const KeyType & x)
{
	if(isLeaf()) return this;
	
	BNode * n = leftTo(x);
	return n->leafLeftTo(x);
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::sendDataRight(int num, BNode * rgt)
{
	for(int i = 0; i < num; i++) {
		rgt->insertData(KeyNData<KeyType, MaxNKey>::lastData());
		KeyNData<KeyType, MaxNKey>::removeLastData();
	}
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::sendDataLeft(int num, BNode * lft)
{
/// if send all in leaf, connect sibling to lft
    if(num == KeyNData<KeyType, MaxNKey>::numKeys() && isLeaf()) {
		Entity * rgt = sibling();
		lft->connectSibling(rgt);
	}
    
	for(int i = 0; i < num; i++) {
		lft->insertData(KeyNData<KeyType, MaxNKey>::firstData());
		KeyNData<KeyType, MaxNKey>::removeFirstData();
	}
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::removeLeaf(const KeyType & x)
{
	if(!removeDataLeaf(x)) {
		//std::cout<<"\n error bnode cannot remove data "<<x
		//	<<" in "<<str();
		return;
	}
	
	//std::cout<<"\n bnode remove in leaf "<<x
	//	<<" n key "<<KeyNData<KeyType, MaxNKey>::numKeys();
	//if(!isUnderflow()) return;

	if(!leafBalance())
	    mergeLeaf();
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::mergeLeaf()
{
	if( mergeLeafRight() ) return true;
	return mergeLeafLeft();
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::mergeLeafRight()
{
	BNode * rgt = siblingNode();
/// must have sibling
	if(!rgt) return false;
	
	BNode * up = siblingNode()->parentNode();
/// must share parent
	if(parentNode() != up) return false;
	
	if(!shouldLeafMerge(this, siblingNode() )) return false;
	
#if 0
	std::cout<<"\n merge "<<str()<<" + "<<rgt->str();
#endif
	 
	Pair<KeyType, Entity> rgt1st = rgt->firstData();

	siblingNode()->sendDataLeft(siblingNode()->numKeys(), this);

	delete rgt;
	
    Pair<KeyType, Entity> k = up->dataRightTo(rgt1st.key);
	
	k.index = this;
	int ki = up->keyRight(k.key);

	up->setData(ki, k);
	up->pop(k);
	
	return true;
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::mergeLeafLeft()
{
	const Pair<KeyType, Entity> s = KeyNData<KeyType, MaxNKey>::firstData();
/// must share parent
	if(!parentNode()->hasKey(s.key)) return false;
	BNode * leftSibling = parentNode()->leafLeftTo(s.key);
	return leftSibling->mergeLeafRight();
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::removeDataLeaf(const KeyType & x)
{
    SearchResult s = findKey(x);
	if(s.found < 0) return false;
	
	int found = s.found;
    
/// remove leaf connection
	if(KeyNData<KeyType, MaxNKey>::index(found) ) {
		delete KeyNData<KeyType, MaxNKey>::index(found);
		KeyNData<KeyType, MaxNKey>::dataR(found)->index = NULL;
	}

/// last is easy
	if(found == KeyNData<KeyType, MaxNKey>::numKeys() - 1) {
		KeyNData<KeyType, MaxNKey>::reduceNumKeys();//std::cout<<"reduce last in leaf to "<<numKeys();
		return true;
	}
	
	for(int i= found; i < KeyNData<KeyType, MaxNKey>::numKeys() - 1; i++)
        *KeyNData<KeyType, MaxNKey>::dataR(i) = KeyNData<KeyType, MaxNKey>::data(i+1);
		
	if(found == 0) {
	    bool c = false;
		BNode * crossed = ancestor(x, c);
		if(c) crossed->replaceKey(x, KeyNData<KeyType, MaxNKey>::firstData().key);
	}
		
    KeyNData<KeyType, MaxNKey>::reduceNumKeys();//std::cout<<"reduce in leaf to "<<numKeys();
    return true;
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::behead()
{
	connectSibling(KeyNData<KeyType, MaxNKey>::index(0) );
	KeyNData<KeyType, MaxNKey>::removeFirstData1();
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::removeKey(const KeyType & x)
{
	SearchResult s = findKey(x);
	
	if(s.found < 0) return false;
	
#if 0
	std::cout<<"\n rm k["<<s.found<<"] "<<x;
#endif

	int found = s.found;
	
	if(found == 0) {
	    connectLeftChild(KeyNData<KeyType, MaxNKey>::index(found) );
	}
	else {
		// m_data([found - 1].index = m_data[found].index;
		KeyNData<KeyType, MaxNKey>::setIndex(found - 1, KeyNData<KeyType, MaxNKey>::index(found) );
	}

	if(found == KeyNData<KeyType, MaxNKey>::numKeys() - 1) {
		KeyNData<KeyType, MaxNKey>::reduceNumKeys();
		return true;
	}
	
	for(int i= found; i < KeyNData<KeyType, MaxNKey>::numKeys() - 1; i++)
		KeyNData<KeyType, MaxNKey>::setData(i, KeyNData<KeyType, MaxNKey>::data(i+1) );
		
    KeyNData<KeyType, MaxNKey>::reduceNumKeys();
	return true;
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::pop(const Pair<KeyType, Entity> & x)
{
	if(isRoot() ) popRoot(x);
	else popInterior(x);
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::popRoot(const Pair<KeyType, Entity> & x)
{
#if 0
	std::cout<<"\n pop "<<str()<<" by "<<x.key;
#endif	
	if(KeyNData<KeyType, MaxNKey>::isSingular() ) {
	    if(hasChildren() )
	        takeover();
	    else 
	        removeKey(x.key);
	}
	else 
	    removeKey(x.key);
#if 0	
	std::cout<<"\n aft pop "<<str();
#endif
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::popInterior(const Pair<KeyType, Entity> & x)
{
#if 0
	std::cout<<"\n pop "<<str()<<" by "<<x.key;
#endif	
	if(x.key == KeyNData<KeyType, MaxNKey>::firstKey() ) {
	    KeyNData<KeyType, MaxNKey>::removeFirstData1();
	}
	else removeKey(x.key);
#if 0	
	std::cout<<"\n aft ";
	dbgLinks(false);
#endif	
	//if(!isUnderflow()) return;
	
	mergeInterior(x.key);
	
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::mergeData(BNode * another, int start)
{
	const int num = another->numKeys();
	for(int i = start; i < num; i++)
		insertData(another->data(i));
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::mergeInterior(const KeyType x)
{
	if(mergeInteriorRight(x)) return true;
	return mergeInteriorLeft(x);
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::mergeInteriorRight(const KeyType x)
{
    int kr;
    BNode * rgt = rightInteriorNeighbor(kr);
	
    if(!rgt) return false;
    
#if 0
	std::cout<<"\n  merge "<<str()
	    <<" <- "<<rgt->str()
		<<" by "<<x;
#endif

	if(!shouldInteriorMerge(this, rgt)) {
#if 0
	    std::cout<<"\n  cannot merge "<<str()
				<<" with "<<rgt->str();
#endif
	    return false;
	}
	
	Pair<KeyType, Entity> k;
	k.key = parentNode()->key(kr);
	k.index = rgt->leftChild();
	
	insertData(k);
    
	rgt->sendDataLeft(rgt->numKeys(), this);
	
#if 0    
	std::cout<<"\n aft "<<str();
#endif	

	connectToChildren();
	
	k.key = rgt->firstKey();
	k.index = this;
	parentNode()->setData(kr, k);
	parentNode()->pop(k);
	
	return true;
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::mergeInteriorLeft(const KeyType x)
{
	BNode * lft =leftInteriorNeighbor();
	if(!lft) return false;
	
	return lft->mergeInteriorRight(x);
}

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::rightInteriorNeighbor(int & indParent) const
{
    if(parentNode()->lastKey() < KeyNData<KeyType, MaxNKey>::firstKey() ) return NULL;
    if(parentNode()->firstKey() >= KeyNData<KeyType, MaxNKey>::lastKey() ) {
        indParent = 0;
        return parentNode()->linkedNode(0);
    }
    
    SearchResult s = parentNode()->findKey(KeyNData<KeyType, MaxNKey>::firstKey() );
	if(s.found > -1) indParent = s.found;
	else indParent = s.high;
	
	// std::cout<<"\n right inner nei kr "<<indParent;
	
	return parentNode()->linkedNode(indParent);
}

/// parent.index[k] -> key[0]
/// j <- k-1
/// parent.index[j] -> left_neighbor.key[0]
template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::leftInteriorNeighbor() const
{
    if(parentNode()->firstKey() > KeyNData<KeyType, MaxNKey>::lastKey() ) return NULL;
    if(parentNode()->index(0) == this) return parentNode()->leftChildNode();
    
	Pair<KeyType, Entity> k, j;
	if(!parentNode()->dataLeftTo(KeyNData<KeyType, MaxNKey>::firstKey(), k)) return NULL;
	j = k;
	if(j.index == this)
		parentNode()->dataLeftTo(k.key, j);
	
	if(j.index == this) return NULL;
	return static_cast<BNode *>(j.index);
}

template <typename KeyType, int MaxNKey> 
bool BNode<KeyType, MaxNKey>::dataLeftTo(const KeyType & x, Pair<KeyType, Entity> & dst) const
{
	int i = keyLeft(x);
	if(i > -1) {
		dst = KeyNData<KeyType, MaxNKey>::data(i);
		return true;
	}
	dst.index = leftChild();
	return false;
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::dbgUp()
{
    int kr = parentNode()->keyLeft(KeyNData<KeyType, MaxNKey>::firstKey() );
    if(kr < 0) kr = 0;
	std::cout<<"\n parent "<<parentNode()->str()
	    <<"\n k "<<kr
	    <<"\n  "<<str();
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::dbgRightUp(BNode<KeyType, MaxNKey> * rgt)
{
    int kr = parentNode()->keyLeft(rgt->firstKey() );
    if(kr < 0) kr = 0;
	std::cout<<"\n parent "<<parentNode()->str()
	    <<"\n k "<<kr
	    <<"\n  "<<str()
	    <<" <- "<<rgt->str();
}

template <typename KeyType, int MaxNKey>
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::linkedNode(int idx) const
{ return static_cast<BNode *>(KeyNData<KeyType, MaxNKey>::index(idx) ); }

template <typename KeyType, int MaxNKey>
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::leftChildNode() const
{ return static_cast<BNode *>(leftChild() ); }

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::parentNode() const
{ return static_cast<BNode *>(parent()); }

template <typename KeyType, int MaxNKey> 
BNode<KeyType, MaxNKey> * BNode<KeyType, MaxNKey>::siblingNode() const
{ return static_cast<BNode *>(sibling()); }

template <typename KeyType, int MaxNKey> 
const std::string BNode<KeyType, MaxNKey>::str() const 
{
	std::stringstream sst;
	if(isRoot() ) sst<<" root  ";
	else if(isLeaf() ) sst<<" leaf  ";
	else sst<<" inner ";
	
	int i = 0;
	sst<<" n "<<KeyNData<KeyType, MaxNKey>::numKeys()<<" [";
    if(KeyNData<KeyType, MaxNKey>::numKeys() ==0) {
        sst<<"] ";
    } else {
    
    for(;i<KeyNData<KeyType, MaxNKey>::numKeys()-1;++i) 
		sst<<KeyNData<KeyType, MaxNKey>::key(i)<<",";
		
    sst<<KeyNData<KeyType, MaxNKey>::lastKey()<<"] ";
    }
	
	if(!siblingNode() ) sst<<"~";
	return sst.str();
}

template <typename KeyType, int MaxNKey> 
void BNode<KeyType, MaxNKey>::dbgDown() const 
{
	int i;
	if(isLeaf() ) {
		if(siblingNode() ) std::cout<<" | "<<siblingNode()->firstKey();
		else std::cout<<"~";
	}
	else {
		if(leftChildNode() ) std::cout << "\n   " << leftChildNode()->firstKey() << " < ";
		for(i=0;i<KeyNData<KeyType, MaxNKey>::numKeys();++i)
		std::cout << " > " << linkedNode(i)->firstKey();
	}
}

template <typename KeyType, int MaxNKey> 
Pair<Entity *, Entity> BNode<KeyType, MaxNKey>::find(const KeyType & x)
{
	if(isRoot()) 
		return findRoot(x);
	else if(isLeaf())
		return findLeaf(x);
	
	return findInterior(x);
}

template <typename KeyType, int MaxNKey> 
Pair<Entity *, Entity> BNode<KeyType, MaxNKey>::findRoot(const KeyType & x)
{
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		return n->find(x);
	}
	return findLeaf(x);
}

template <typename KeyType, int MaxNKey> 
Pair<Entity *, Entity> BNode<KeyType, MaxNKey>::findLeaf(const KeyType & x)
{
	Pair<Entity *, Entity> r;
	r.key = this;
	r.index = NULL;
	if(!isKeyInRange(x) ) {
		return r;
	}
	int f = findKey(x).found;
	if(f < 0) return r;
	
	r.index = KeyNData<KeyType, MaxNKey>::index(f);
	
	return r;
}

template <typename KeyType, int MaxNKey> 
Pair<Entity *, Entity> BNode<KeyType, MaxNKey>::findInterior(const KeyType & x)
{
	BNode * n = nextIndex(x);
	return n->find(x);
}

template <typename KeyType, int MaxNKey>
Pair<Entity *, Entity> BNode<KeyType, MaxNKey>::findInNode(const KeyType & x, SearchResult * result)
{ 
	Pair<Entity *, Entity> r;
	r.key = this;
	r.index = NULL;
	if(!isKeyInRange(x)) {
		result->found = -1;
		result->low = 0;
		result->high = KeyNData<KeyType, MaxNKey>::numKeys() - 1;
		if(KeyNData<KeyType, MaxNKey>::firstKey() > x) 
			r.index = KeyNData<KeyType, MaxNKey>::firstData().index;
		else 
			r.index = KeyNData<KeyType, MaxNKey>::lastData().index;
		return r;
	}
	
	*result = findKey(x);

	if(result->found < 0) return r;
	
	r.index = KeyNData<KeyType, MaxNKey>::index(result->found);
	
	return r; 
}

template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::dbgFind(const KeyType & x)
{
	if(isRoot()) 
		return dbgFindInRoot(x);
	else if(isLeaf())
		return dbgFindInLeaf(x);
	
	return dbgFindInInterior(x);
}

template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::dbgFindInRoot(const KeyType & x)
{
	std::cout<<"\n dbg root "<<str();
	if(hasChildren()) {
		BNode * n = nextIndex(x);
		return n->dbgFind(x);
	}
	return dbgFindInLeaf(x);
}

template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::dbgFindInLeaf(const KeyType & x)
{
	std::cout<<"\n dbg leaf "<<str();
	if(!isKeyInRange(x) ) {
		std::cout<<"\n out of range";
		return;
	}
	int found = findKey(x).found;
	
	if(found < 0) {
		std::cout<<"\n not found";
		return;
	}
	
	std::cout<<"\n found "<<x<<" key["<<found<<"]";
}

template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::dbgFindInInterior(const KeyType & x)
{
	std::cout<<"\n dbg inner "<<str();
	BNode * n = nextIndex(x);
	return n->dbgFind(x);
}

template <typename KeyType, int MaxNKey>
bool BNode<KeyType, MaxNKey>::dbgLinks(bool silient)
{	
    if(!hasChildren()) return true;
	bool stat = true;
	if(!silient) std::cout<<"\n n k "<<KeyNData<KeyType, MaxNKey>::numKeys();
	KeyType pre = leftChildNode()->firstKey();
	if(!silient) std::cout<<" "<<pre<<" | ";
	
	for(int i=0;i<KeyNData<KeyType, MaxNKey>::numKeys(); ++i) {
	
		KeyType cur = linkedNode(i)->firstKey();
		if(!silient) std::cout<<"  "<<cur;
		if(cur <= pre) {
			std::cout<<"\n\n****    error wrong link "
			        <<pre<<" >= "<<cur;
			stat = false;
		}
		pre = cur;
	}
	
	for(int i=0;i<KeyNData<KeyType, MaxNKey>::numKeys(); ++i) {
		if(KeyNData<KeyType, MaxNKey>::key(i) > linkedNode(i)->firstKey() ) {
			//if(!silient) 
				std::cout<<"\n\n****    error wrong k["<<i<<"]"
						<<KeyNData<KeyType, MaxNKey>::key(i)
						<<" > "
						<<linkedNode(i)->str()
						;
			stat = false;
		}
	}
	
	if(isLeaf() ) {
		if(siblingNode() ) {
			if(KeyNData<KeyType, MaxNKey>::lastKey() >= siblingNode()->firstKey() ) {
				//if(!silient) 
					std::cout<<"\n\n****    error order to sibling "
							<<KeyNData<KeyType, MaxNKey>::lastKey()
							<<" > "
							<<siblingNode()->firstKey()
							;
				stat = false;
			}
		}
	}
	else if(leftChild() ) {
		if(KeyNData<KeyType, MaxNKey>::firstKey() < leftChildNode()->lastKey() ) {
				if(!silient) std::cout<<"\n\n****    error order to left child "
							<<KeyNData<KeyType, MaxNKey>::firstKey()
							<<" < ["<<leftChildNode()->firstKey()
							<<", "<<leftChildNode()->lastKey()<<"] "
							;
				stat = false;
			}
	}
	if(!stat) {
	    std::cout<<"\n failed link check\n"<<str();
	    dbgDown();
	}
	return stat;
}

template <typename KeyType, int MaxNKey>
int BNode<KeyType, MaxNKey>::countBalance(int a, int b)
{ return (a + b)/ 2 - a;}

template <typename KeyType, int MaxNKey>
bool BNode<KeyType, MaxNKey>::shouldLeafMerge(BNode<KeyType, MaxNKey> * lft, BNode<KeyType, MaxNKey> * rgt) const 
{ return (lft->numKeys() + rgt->numKeys()) <= MaxNKey; }

template <typename KeyType, int MaxNKey>
bool BNode<KeyType, MaxNKey>::shouldInteriorMerge(BNode<KeyType, MaxNKey> * lft, BNode<KeyType, MaxNKey> * rgt) const 
{ return (lft->numKeys() + rgt->numKeys()) < MaxNKey; }
	
template <typename KeyType, int MaxNKey>
int BNode<KeyType, MaxNKey>::shouldBalance(BNode<KeyType, MaxNKey> * lft, BNode<KeyType, MaxNKey> * rgt) const 
{ return (lft->numKeys() + rgt->numKeys()) / 2 - lft->numKeys(); }

template <typename KeyType, int MaxNKey>
KeyType BNode<KeyType, MaxNKey>::getHighestKey() const
{
	if(isLeaf() ) return KeyNData<KeyType, MaxNKey>::lastKey();
	return linkedNode(KeyNData<KeyType, MaxNKey>::numKeys() - 1)->getHighestKey();
}

template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::takeover()
{    
    if(leftChildNode()->isLeaf() )
        leafTakeover();
    else
        innerTakeover();
}
	
template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::leafTakeover()
{
    BNode * lft = leftChildNode();
    int i, j=0;
    for(i = 0; i< lft->numKeys(); ++i) 
        setData(j++, lft->data(i) );
    KeyNData<KeyType, MaxNKey>::setNumKeys(j);
    connectSibling(NULL);
}

template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::innerTakeover()
{
    BNode * lft = leftChildNode();
    BNode * lftLnk = lft->leftChildNode();
    
    int i, j=0;
    for(i = 0; i< lft->numKeys(); ++i) 
        setData(j++, lft->data(i) );
    
    KeyNData<KeyType, MaxNKey>::setNumKeys(j);
    connectSibling(lftLnk);
	connectToChildren();
}

template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::compressSingular()
{
    if(!KeyNData<KeyType, MaxNKey>::isSingular() ) return;
	
    if(!leftChildNode() ) return;
	if(leftChildNode()->numKeys() 
		+ linkedNode(0)->numKeys() > MaxNKey - 1) return;
		
	if(leftChildNode()->isLeaf() )
        leaf3WayMerge();
    else
        inner3WayMerge();
	
}
	
template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::leaf3WayMerge()
{
    BNode * lft = leftChildNode();
    BNode * rgt = linkedNode(0);
    int i, j=0;
    for(i = 0; i< lft->numKeys(); ++i) 
        setData(j++, lft->data(i) );
    for(i = 0; i< rgt->numKeys(); ++i) 
        setData(j++, rgt->data(i) );
    KeyNData<KeyType, MaxNKey>::setNumKeys(j);
    connectSibling(NULL);
}

template <typename KeyType, int MaxNKey>
void BNode<KeyType, MaxNKey>::inner3WayMerge()
{
    BNode * lft = leftChildNode();
    BNode * lftLnk = lft->leftChildNode();
    BNode * rgt = linkedNode(0);
    BNode * rgtLnk = rgt->leftChildNode();
    
    Pair<KeyType, Entity> k;
    k.key = rgtLnk->firstKey();
    k.index = rgtLnk;
    
    int i, j=0;
    for(i = 0; i< lft->numKeys(); ++i) 
        setData(j++, lft->data(i) );
    
    setData(j++, k);
    
    for(i = 0; i< rgt->numKeys(); ++i) 
        setData(j++, rgt->data(i) );
    
    KeyNData<KeyType, MaxNKey>::setNumKeys(j);
    connectSibling(lftLnk);
	connectToChildren();
}

template <typename KeyType, int MaxNKey>
bool BNode<KeyType, MaxNKey>::isUnderflow() const
{ return KeyNData<KeyType, MaxNKey>::numKeys() < (MaxNKey>>1); }

} // end of namespace sdb

}