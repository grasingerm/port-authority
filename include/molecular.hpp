#ifndef __MOLECULAR_HPP__
#define __MOLECULAR_HPP__

#include "pauth_types.hpp"
#include "unordered_pair.hpp"
#include <unordered_map>

namespace pauth {

enum class molecular_id { Ar, Cu, H, O, C, Test, Test1, Test2 };
using molecular_name_map = std::unordered_map<std::string, molecular_id>;

extern const molecular_name_map molecular_names_to_ids;

struct molecular_pair_hash {
  std::size_t operator()(const unordered_pair<molecular_id> &k) const {
    return ((std::hash<int>()(static_cast<int>(k.first))) ^ 
            (std::hash<int>()(static_cast<int>(k.second))));
  }
};

using molecular_pair_interaction_map =
    std::unordered_map<unordered_pair<molecular_id>, double, molecular_pair_hash>;

}

#endif
