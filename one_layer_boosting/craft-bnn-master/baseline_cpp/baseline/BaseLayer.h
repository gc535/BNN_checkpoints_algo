//------------------------------------------------------------------------
// Base class for layers
//------------------------------------------------------------------------
#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include <string>
#include <sstream>
#include "SArray.h"
#include "Debug.h"

template<unsigned M, unsigned N,
         unsigned SIZE_I, unsigned SIZE_O>
class SquareLayer {
  
public:

  //--------------------------------------------------
  // Static parameters
  static constexpr unsigned num_inputs() { return M; }
  static constexpr unsigned num_units() { return N; }
  static constexpr unsigned input_size() { return SIZE_I; }
  static constexpr unsigned output_size() { return SIZE_O; }
  
  //--------------------------------------------------
  // Constructor
  SquareLayer(std::string name)
  {
    std::cout << name << ":\n";
    std::cout << dump();
  }

  std::string dump() const {
    std::ostringstream ss;
    ss << "\tLayer Size:  " << num_inputs() << " x " << num_units() << "\n";
    ss << "\tOutput Size: " << output_size() << "\n";
    return ss.str();
  }

};

#endif
