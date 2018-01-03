//------------------------------------------------------------------------
// Simple vector of bits, 8 bits packed into 1 byte
//------------------------------------------------------------------------
#ifndef BIT_VECTOR_H
#define BIT_VECTOR_H

#include <assert.h>
#include <cstdint>


struct BitVector {
  // Underlying storage type
  typedef uint8_t ElemType;
  static const unsigned ELEM_BITS = sizeof(ElemType)*8;
  
  unsigned m_len;   // number of bits stored
  ElemType* m_data;

  // Constructor
  BitVector(unsigned len)
    : m_len(len)
  {
    assert(len % ELEM_BITS == 0);
    m_data = new ElemType[m_len/(ELEM_BITS)];
  }

  // Destructor
  ~BitVector() {
    delete[] m_data;
  }

  // Set
  void set(unsigned i) {
    unsigned idx = i/ELEM_BITS;
    ElemType mask = 1 << i%(ELEM_BITS);
    m_data[idx] |= mask;
  }
  
  // Unset
  void unset(unsigned i) {
    unsigned idx = i/ELEM_BITS;
    ElemType mask = ~(1 << i%(ELEM_BITS));
    m_data[idx] &= mask;
  }

  // Get
  uint8_t get(unsigned i) const {
    unsigned idx = i/ELEM_BITS;
    ElemType mask = 1 << i%(ELEM_BITS);
    return m_data[idx] & mask;
  }
  
  // operator[]
  uint8_t operator[](unsigned i) const { return get(i); }

  unsigned size() const { return m_len; }
  unsigned bytesize() const { return m_len / ELEM_BITS; }
  ElemType* data() const { return m_data; }
};

#endif
