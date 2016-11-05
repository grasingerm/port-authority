#ifndef __MOLECULAR_HPP__
#define __MOLECULAR_HPP__

#include "unordered_pair.hpp"
#include <unordered_map>

namespace mmd {

enum class molecular_id { Ar, Cu, Test, Test1, Test2 };
using molecular_name_map = std::unordered_map<std::string, molecular_id>;
using molecular_pair_interaction_map =
    std::unordered_map<unordered_pair<molecular_id>, double>;

extern const molecular_name_map molecular_names_to_ids;
}

#endif
