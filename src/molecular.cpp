#include "molecular.hpp"

namespace pauth {

const molecular_name_map
    molecular_names_to_ids({{"Argon", pauth::molecular_id::Ar},
                            {"Ar", pauth::molecular_id::Ar},
                            {"Copper", pauth::molecular_id::Cu},
                            {"Cu", pauth::molecular_id::Cu},
                            {"Hydrogen", pauth::molecular_id::H},
                            {"H", pauth::molecular_id::H},
                            {"Oyxgen", pauth::molecular_id::O},
                            {"O", pauth::molecular_id::O},
                            {"Carbon", pauth::molecular_id::C},
                            {"C", pauth::molecular_id::C},
                            {"Test", pauth::molecular_id::Test},
                            {"Test1", pauth::molecular_id::Test1},
                            {"Test2", pauth::molecular_id::Test2}});

}
