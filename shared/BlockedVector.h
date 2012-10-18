/*
 *  BlockedVector.h
 *  kdtree
 *
 *  Created by jian zhang on 10/19/12.
 *  Copyright 2012 __MyCompanyName__. All rights reserved.
 *
 */
#include <vector>
template <typename T, size_t BlockSize> class BlockedVector {
public:
	BlockedVector() : m_pos(0) {}

	~BlockedVector() {
		clear();
	}

	/**
	 * \brief Append an element to the end
	 */
	inline void push_back(const T &value) {
		size_t blockIdx = m_pos / BlockSize;
		size_t offset = m_pos % BlockSize;
		if (blockIdx == m_blocks.size())
			m_blocks.push_back(new T[BlockSize]);
		m_blocks[blockIdx][offset] = value;
		m_pos++;
	}

	/**
	 * \brief Allocate a certain number of elements and
	 * return a pointer to the first one.
	 *
	 * The implementation will ensure that they lie
	 * contiguous in memory -- note that this can potentially
	 * create unused elements in the previous block if a new
	 * one has to be allocated.
	 */
	inline T * allocate(size_t size) {
		size_t blockIdx = m_pos / BlockSize;
		size_t offset = m_pos % BlockSize;
		T *result;
		if (offset + size <= BlockSize) {
			if (blockIdx == m_blocks.size())
				m_blocks.push_back(new T[BlockSize]);
			result = m_blocks[blockIdx] + offset;
			m_pos += size;
		} else {
			++blockIdx;
			if (blockIdx == m_blocks.size())
				m_blocks.push_back(new T[BlockSize]);
			result = m_blocks[blockIdx];
			m_pos += BlockSize - offset + size;
		}
		return result;
	}

	inline T &operator[](size_t index) {
		return *(m_blocks[index / BlockSize] +
			(index % BlockSize));
	}

	inline const T &operator[](size_t index) const {
		return *(m_blocks[index / BlockSize] +
			(index % BlockSize));
	}


	/**
	 * \brief Return the currently used number of items
	 */
	inline size_t size() const {
		return m_pos;
	}

	/**
	 * \brief Return the number of allocated blocks
	 */
	inline size_t blockCount() const {
		return m_blocks.size();
	}

	/**
	 * \brief Return the total capacity
	 */
	inline size_t capacity() const {
		return m_blocks.size() * BlockSize;
	}

	/**
	 * \brief Resize the vector to the given size.
	 *
	 * Note: this implementation doesn't support 
	 * enlarging the vector and simply changes the
	 * last item pointer.
	 */
	inline void resize(size_t pos) {
		m_pos = pos;
	}

	/**
	 * \brief Release all memory
	 */
	void clear() {
		for (typename std::vector<T *>::iterator it = m_blocks.begin(); 
				it != m_blocks.end(); ++it)
			delete[] *it;
		m_blocks.clear();
		m_pos = 0;
	}
private:
	std::vector<T *> m_blocks;
	size_t m_pos;
};