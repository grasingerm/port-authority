#ifndef __CALLBACKS_HPP__
#define __CALLBACKS_HPP__

#include "mprof.hpp"
#include "simulation.hpp"
#include <armadillo>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

/* TODO: can we create a callback that caches values, then callbacks that read
 *       the cache?
 */

namespace mmd {

/*! \brief Wrapper for accessing molecular positions
 *
 * \param   sim   Simulation object
 * \return        Molecular positions
 */
inline const arma::mat &positions(const simulation &sim) {
  return sim.get_positions();
}

/*! \brief Wrapper for accessing molecular velocities
 *
 * \param   sim   Simulation object
 * \return        Molecular velocities
 */
inline const arma::mat &velocities(const simulation &sim) {
  return sim.get_velocities();
}

/*! \brief Wrapper for accessing intermolecular forces
 *
 * \param   sim   Simulation object
 * \return        Molecular forces
 */
inline const arma::mat &forces(const simulation &sim) {
  return sim.get_forces();
}

/*! \brief Wrapper for accessing current simulation time
 *
 * \param   sim   Simulation object
 * \return        Current time
 */
inline double time(const simulation &sim) { return sim.get_time(); }

/*! \brief Callback for saving xyz data to file
 */
class save_xyz_callback {
public:
  /*! \brief Constructor for callback function that saves data in xyz format
   *
   * \param   fname       File name of the output file
   * \param   dt          Frequency with which to write data
   * \param   da          Function for accessing simulation data
   * \param   prepend_id  Flag, if true prepend each row with molecular id
   * \return          Callback function
   */
  save_xyz_callback(const char *fname, const double dt, data_accessor da,
                    const bool prepend_id=true)
      : fname(fname), outfile(fname), dt(dt), da(da), prepend_id(prepend_id) {}

  /*! \brief Constructor for callback function that saves data in xyz format
   *
   * \param   fname       File name of the output file
   * \param   dt          Frequency with which to write data
   * \param   da          Function for accessing simulation data
   * \param   prepend_id  Flag, if true prepend each row with molecular id
   * \return          Callback function
   */
  save_xyz_callback(const std::string &fname, const double dt, data_accessor da,
                    const bool prepend_id=true)
      : fname(fname), outfile(fname), dt(dt), da(da), prepend_id(prepend_id) {}

  /*! \brief Copy constructor for callback function that saves data in xyz
   * format
   *
   * \param   cb      Callback function
   * \return          Callback function
   */
  save_xyz_callback(const save_xyz_callback &cb)
      : fname(cb.fname), outfile(cb.fname), dt(cb.dt), da(cb.da), 
        prepend_id(cb.prepend_id) {}

  ~save_xyz_callback() { outfile.close(); }

  void operator()(const simulation &);

  /*! \brief Get filename of the output file
   *
   * \return    Filename of the output file
   */
  inline const std::string &get_fname() const { return fname; }

private:
  std::string fname;
  std::ofstream outfile;
  double dt;
  data_accessor da;
  bool prepend_id;
};

/*! \brief Callback for saving data values to a delimited file
 */
class save_values_with_time_callback {
public:
  /*! \brief Constructor for callback function that saves delimited data
   *
   * \param   fname   File name of the output file
   * \param   dt      Frequency with which to write data
   * \param   vas     Functions for accessing simulation values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  save_values_with_time_callback(
      const char *fname, const double dt,
      const std::initializer_list<value_accessor> &vas, const char delim = ',')
      : fname(fname), outfile(fname), dt(dt), vas(vas), delim(delim) {
    if (dt < 0)
      throw std::invalid_argument("Time between callbacks, dt, "
                                  "must be positive");
  }

  /*! \brief Constructor for callback function that saves delimited data
   *
   * \param   fname   File name of the output file
   * \param   dt      Frequency with which to write data
   * \param   vas     Functions for accessing simulation values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  save_values_with_time_callback(
      const std::string &fname, const double dt,
      const std::initializer_list<value_accessor> &vas, const char delim = ',')
      : fname(fname), outfile(fname), dt(dt), vas(vas), delim(delim) {
    if (dt < 0)
      throw std::invalid_argument("Time between callbacks, dt, "
                                  "must be positive");
  }

  /*! \brief Copy constructor
   *
   * \param   cb      Callback function to copy
   * \return          Callback function
   */
  save_values_with_time_callback(const save_values_with_time_callback &cb)
      : fname(cb.fname), outfile(cb.fname), dt(cb.dt), vas(cb.vas),
        delim(cb.delim) {}

  ~save_values_with_time_callback() { outfile.close(); }

  void operator()(const simulation &);

  /*! \brief Get filename of the output file
   *
   * \return    Filename of the output file
   */
  inline const std::string &get_fname() const { return fname; }

private:
  std::string fname;
  std::ofstream outfile;
  double dt;
  std::vector<value_accessor> vas;
  char delim;
};

/*! \brief Callback for saving data values to a delimited file
 */
template <size_t N> class save_vector_with_time_callback {
public:
  /*! \brief Constructor for callback function that saves delimited data
   *
   * \param   fname   File name of the output file
   * \param   dt      Frequency with which to write data
   * \param   ras     Row accessor
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  save_vector_with_time_callback<N>(
      const char *fname, const double dt,
      std::function<std::array<double, N>(const simulation&)> ras, 
      const char delim = ',')
      : fname(fname), outfile(fname), dt(dt), ras(ras), delim(delim) {
    if (dt < 0)
      throw std::invalid_argument("Time between callbacks, dt, "
                                  "must be positive");
  }

  /*! \brief Constructor for callback function that saves delimited data
   *
   * \param   fname   File name of the output file
   * \param   dt      Frequency with which to write data
   * \param   vas     Functions for accessing simulation values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  save_vector_with_time_callback<N>(
      const std::string &fname, const double dt,
      std::function<std::array<double, N>(const simulation&)> ras, 
      const char delim = ',')
      : fname(fname), outfile(fname), dt(dt), ras(ras), delim(delim) {
    if (dt < 0)
      throw std::invalid_argument("Time between callbacks, dt, "
                                  "must be positive");
  }

  /*! \brief Copy constructor
   *
   * \param   cb      Callback function to copy
   * \return          Callback function
   */
  save_vector_with_time_callback<N>(const save_vector_with_time_callback<N> &cb)
      : fname(cb.fname), outfile(cb.fname), dt(cb.dt), ras(cb.ras),
        delim(cb.delim) {}

  ~save_vector_with_time_callback<N>() { outfile.close(); }

  void operator()(const simulation &sim) {
    if (fmod(sim.get_time(), dt) < sim.get_dt()) {
      outfile << sim.get_time();
      const auto &values = ras(sim);
      for (const auto &value : values) outfile << delim << value;
      outfile << '\n';
    }
  }

  /*! \brief Get filename of the output file
   *
   * \return    Filename of the output file
   */
  inline const std::string &get_fname() const { return fname; }

private:
  std::string fname;
  std::ofstream outfile;
  double dt;
  std::function<std::array<double, N>(const simulation&)> ras;
  char delim;
};

/*! Factory for constructing an energy and momentum saving callback
 *
 * \param   fname   File name of the output file
 * \param   dt      Frequency with which to write data
 * \param   delim   Character delimiter
 * \return          Callback function
 */
inline save_values_with_time_callback
save_energy_and_momentum_with_time_callback(const char *fname, const double dt,
                                            const char delim = ',') {

  return save_values_with_time_callback(
      fname, dt, {potential_energy, kinetic_energy, total_energy,
                  [](const simulation &sim) -> double {
                    return arma::norm(momentum(sim), 2);
                  }},
      delim);
}

/*! Factory for constructing an energy and momentum saving callback
 *
 * \param   fname   File name of the output file
 * \param   dt      Frequency with which to write data
 * \param   delim   Character delimiter
 * \return          Callback function
 */
inline save_values_with_time_callback
save_energy_and_momentum_with_time_callback(const std::string &fname,
                                            const double dt,
                                            const char delim = ',') {

  return save_values_with_time_callback(
      fname, dt, {potential_energy, kinetic_energy, total_energy,
                  [](const simulation &sim) -> double {
                    return arma::norm(momentum(sim), 2);
                  }},
      delim);
}

/*! \brief Callback for printing data values to a delimited file
 */
class print_values_with_time_callback {
public:
  /*! \brief Constructor for callback function that prints data
   *
   * \param   ostr    Output stream
   * \param   dt      Frequency with which to write data
   * \param   vas     Functions for accessing simulation values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  print_values_with_time_callback(
      std::ostream &ostr, const double dt,
      const std::initializer_list<value_accessor> &vas, const char delim = ',')
      : ostr(ostr), dt(dt), vas(vas), delim(delim) {
    if (dt < 0)
      throw std::invalid_argument("Time between callbacks, dt, "
                                  "must be positive");
  }

  /*! \brief Constructor for callback function that prints data
   *
   * \param   ostr    Output stream
   * \param   dt      Frequency with which to write data
   * \param   vas     Functions for accessing simulation values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  print_values_with_time_callback(
      const double dt, const std::initializer_list<value_accessor> &vas,
      const char delim = ' ')
      : ostr(std::cout), dt(dt), vas(vas), delim(delim) {
    if (dt < 0)
      throw std::invalid_argument("Time between callbacks, dt, "
                                  "must be positive");
  }

  /*! \brief Copy constructor
   *
   * \param   cb      Callback function to copy
   * \return          Callback function
   */
  print_values_with_time_callback(const print_values_with_time_callback &cb)
      : ostr(cb.ostr), dt(cb.dt), vas(cb.vas), delim(cb.delim) {}

  ~print_values_with_time_callback() {}

  void operator()(const simulation &);

private:
  std::ostream &ostr;
  double dt;
  std::vector<value_accessor> vas;
  char delim;
};

/*! Factory for constructing an energy and momentum printing callback
 *
 * \param   dt      Frequency with which to write data
 * \param   delim   Character delimiter
 * \return          Callback function
 */
inline print_values_with_time_callback
print_energy_and_momentum_with_time_callback(const double dt,
                                             const char delim = ' ') {

  return print_values_with_time_callback(
      std::cout, dt, {potential_energy, kinetic_energy, total_energy,
                      [](const simulation &sim) -> double {
                        return arma::norm(momentum(sim), 2);
                      }},
      delim);
}

/*! \brief Callback for printing data values to a delimited file
 */
template <size_t N> class print_vector_with_time_callback {
public:
  /*! \brief Constructor for callback function that prints data
   *
   * \param   ostr    Output stream
   * \param   dt      Frequency with which to write data
   * \param   ras     Row accessor
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  print_vector_with_time_callback<N>(
      std::ostream &ostr, const double dt,
      std::function<std::array<double, N>(const simulation&)> ras, 
      const char delim = ',')
      : ostr(ostr), dt(dt), ras(ras), delim(delim) {
    if (dt < 0)
      throw std::invalid_argument("Time between callbacks, dt, "
                                  "must be positive");
  }

  ~print_vector_with_time_callback<N>() {}

  void operator()(const simulation &sim) {
    if (fmod(sim.get_time(), dt) < sim.get_dt()) {
      ostr << sim.get_time();
      const auto &values = ras(sim);
      for (const auto &value : values) ostr << delim << value;
      ostr << '\n';
    }
  }

private:
  std::ostream &ostr;
  double dt;
  std::function<std::array<double, N>(const simulation&)> ras;
  char delim;
};

/*! \brief Check conservation of energy
 *
 * \param   dt  Frequency with which to check energy
 * \param   eps Tolerance of check
 * \return      Callback
 */
callback check_energy(const double dt, const double eps = 1e-6);

/*! \brief Check conservation of momentum
 *
 * \param   dt  Frequency with which to check momentum
 * \param   eps Tolerance of check
 * \return      Callback
 */
callback check_momentum(const double dt, const double eps = 1e-6);

/*! \brief Print current simulation time
 *
 * \param   dt   Frequency with which to print profiling information
 * \param   msg  Message to print
 * \return       Callback
 */
callback print_time(const double dt, const char *msg=""); 

/*! \brief Print time required to simulate dt time
 *
 * \param   dt  Frequency with which to print profiling information
 * \return      Callback
 */
callback print_profile(const double dt);

/*! \brief Outputs xyz data to an output stream in the xyz format
 *
 * \param   ostr        Output stream
 * \param   da          Data accessor for xyz data
 * \param   sim         Simulation object
 * \param   prepend_id  Flag, if true prepend each row with molecular id
 */
void output_xyz(std::ostream &ostr, data_accessor da, const simulation &sim,
                const bool prepend_id=true);

/*! \brief Saves xyz data to a file in the xyz format
 *
 * \param   fname       Filename
 * \param   da          Data accessor for xyz data
 * \param   sim         Simulation object
 * \param   prepend_id  Flag, if true prepend each row with molecular id
 */
inline void save_xyz(const char *fname, data_accessor da, 
                     const simulation &sim, const bool prepend_id=true) {
  std::ofstream outfile(fname, std::ios::out);
  output_xyz(outfile, da, sim, prepend_id);
}

} // namespace mmd

#endif
