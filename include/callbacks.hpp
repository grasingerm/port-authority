#ifndef __CALLBACKS_HPP__
#define __CALLBACKS_HPP__

#include "mprof.hpp"
#include "metropolis.hpp"
#include <armadillo>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>
#include <limits>
#include <map>
#include <iomanip>

/* TODO: can we create a callback that caches values, then callbacks that read
 *       the cache?
 */

namespace pauth {

/*! \brief Callback for saving xyz data to file
 */
class save_xyz_callback {
public:
  /*! \brief Constructor for callback function that saves data in xyz format
   *
   * \param   fname       File name of the output file
   * \param   dstep       Frequency with which to write data
   * \param   da          Function for accessing metropolis data
   * \param   prepend_id  Flag, if true prepend each row with molecular id
   * \return          Callback function
   */
  save_xyz_callback(const char *fname, const unsigned dstep, data_accessor da,
                    const bool prepend_id=true)
      : _fname(fname), _outfile(fname), _dstep(dstep), _da(da), 
        _prepend_id(prepend_id) {}

  /*! \brief Constructor for callback function that saves data in xyz format
   *
   * \param   fname       File name of the output file
   * \param   dstep       Frequency with which to write data
   * \param   da          Function for accessing metropolis data
   * \param   prepend_id  Flag, if true prepend each row with molecular id
   * \return          Callback function
   */
  save_xyz_callback(const std::string &fname, const unsigned dstep, 
                    data_accessor da, const bool prepend_id=true)
      : _fname(fname), _outfile(fname), _dstep(dstep), _da(da), 
        _prepend_id(prepend_id) {}

  /*! \brief Copy constructor for callback function that saves data in xyz
   * format
   *
   * \param   cb      Callback function
   * \return          Callback function
   */
  save_xyz_callback(const save_xyz_callback &cb)
      : _fname(cb._fname), _outfile(cb._fname), _dstep(cb._dstep), _da(cb._da), 
        _prepend_id(cb._prepend_id) {}

  ~save_xyz_callback() { _outfile.close(); }

  void operator()(const metropolis &);

  /*! \brief Get filename of the output file
   *
   * \return    Filename of the output file
   */
  inline const std::string &fname() const { return _fname; }

private:
  std::string _fname;
  std::ofstream _outfile;
  unsigned _dstep;
  data_accessor _da;
  bool _prepend_id;
};

/*! \brief Callback for saving data values to a delimited file
 */
class save_values_with_step_callback {
public:
  /*! \brief Constructor for callback function that saves delimited data
   *
   * \param   fname   File name of the output file
   * \param   dstep   Frequency with which to write data
   * \param   vas     Functions for accessing metropolis values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  save_values_with_step_callback(
      const char *fname, const unsigned dstep,
      const std::initializer_list<value_accessor> &vas, const char delim = ',')
      : _fname(fname), _outfile(fname), _dstep(dstep), _vas(vas), _delim(delim)
  {}

  /*! \brief Constructor for callback function that saves delimited data
   *
   * \param   fname   File name of the output file
   * \param   dstep   Frequency with which to write data
   * \param   vas     Functions for accessing metropolis values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  save_values_with_step_callback(
      const std::string &fname, const unsigned dstep,
      const std::initializer_list<value_accessor> &vas, const char delim = ',')
      : _fname(fname), _outfile(fname), _dstep(dstep), _vas(vas), _delim(delim)
  {}

  /*! \brief Copy constructor
   *
   * \param   cb      Callback function to copy
   * \return          Callback function
   */
  save_values_with_step_callback(const save_values_with_step_callback &cb)
      : _fname(cb._fname), _outfile(cb._fname), _dstep(cb._dstep), _vas(cb._vas),
        _delim(cb._delim) {}

  ~save_values_with_step_callback() { _outfile.close(); }

  void operator()(const metropolis &);

  /*! \brief Get filename of the output file
   *
   * \return    Filename of the output file
   */
  inline const std::string &fname() const { return _fname; }

private:
  std::string _fname;
  std::ofstream _outfile;
  unsigned _dstep;
  std::vector<value_accessor> _vas;
  char _delim;
};

class average_callback {
public:
  /*! Callback for recording averages
   * \param     va              Value accessor
   * \param     dstep_record    Number of steps between recording a value
   * \param     dstep_avg       Number of steps between averageing a value
   */
  average_callback(value_accessor va, const unsigned dstep_record = 1, 
                   const unsigned dstep_avg = 
                   std::numeric_limits<unsigned>::max()) 
    : _sum(0.0), _nsamples(0), _dstep_record(dstep_record), 
      _dstep_avg(dstep_avg), _va(va) {}

  /*! Get averages
   *
   * \return    Averaging data
   */
  inline const auto &avgs() const { return _avgs; }

  /*! Get current average
   *
   * \return    Current average
   */
  inline double current_avg() const { 
    return _sum / static_cast<double>(_nsamples);
  }
  
  void operator()(const metropolis &sim) {
    if (sim.step() % _dstep_record == 0) {  
      _sum += _va(sim);
      ++_nsamples;
    }
    if (sim.step() % _dstep_avg == 0) _avgs[sim.step()] = current_avg();
  }

private:
  long double _sum;
  unsigned long _nsamples;
  unsigned _dstep_record;
  unsigned _dstep_avg;
  value_accessor _va;
  std::map<unsigned long, double> _avgs;
};

/*! \brief Callback for saving data values to a delimited file
 */
template <size_t N> class save_vector_with_step_callback {
public:
  /*! \brief Constructor for callback function that saves delimited data
   *
   * \param   fname   File name of the output file
   * \param   dstep   Frequency with which to write data
   * \param   ras     Row accessor
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  save_vector_with_step_callback<N>(
      const char *fname, const unsigned dstep,
      std::function<std::array<double, N>(const metropolis&)> ras, 
      const char delim = ',')
      : _fname(fname), _outfile(fname), _dstep(dstep), _ras(ras), _delim(delim) 
  {}

  /*! \brief Constructor for callback function that saves delimited data
   *
   * \param   fname   File name of the output file
   * \param   dstep   Frequency with which to write data
   * \param   vas     Functions for accessing metropolis values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  save_vector_with_step_callback<N>(
      const std::string &fname, const unsigned dstep,
      std::function<std::array<double, N>(const metropolis&)> ras, 
      const char delim = ',')
      : _fname(fname), _outfile(fname), _dstep(dstep), _ras(ras), _delim(delim)
  {}

  /*! \brief Copy constructor
   *
   * \param   cb      Callback function to copy
   * \return          Callback function
   */
  save_vector_with_step_callback<N>(const save_vector_with_step_callback<N> &cb)
      : _fname(cb._fname), _outfile(cb._fname), _dstep(cb._dstep), _ras(cb._ras),
        _delim(cb._delim) {}

  ~save_vector_with_step_callback<N>() { _outfile.close(); }

  void operator()(const metropolis &sim) {
    if (sim.step() % _dstep == 0) {
      _outfile << sim.step();
      const auto &values = _ras(sim);
      for (const auto &value : values) _outfile << _delim << value;
      _outfile << '\n';
    }
  }

  /*! \brief Get filename of the output file
   *
   * \return    Filename of the output file
   */
  inline const std::string &fname() const { return _fname; }

private:
  std::string _fname;
  std::ofstream _outfile;
  unsigned _dstep;
  std::function<std::array<double, N>(const metropolis&)> _ras;
  char _delim;
};

/*! \brief Callback for printing data values to a delimited file
 */
class print_values_with_step_callback {
public:
  /*! \brief Constructor for callback function that prints data
   *
   * \param   ostr    Output stream
   * \param   dstep   Frequency with which to write data
   * \param   vas     Functions for accessing metropolis values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  print_values_with_step_callback(
      std::ostream &ostr, const unsigned dstep,
      const std::initializer_list<value_accessor> &vas, const char delim = ',')
      : _ostr(ostr), _dstep(dstep), _vas(vas), _delim(delim)
  {}

  /*! \brief Constructor for callback function that prints data
   *
   * \param   ostr    Output stream
   * \param   dstep   Frequency with which to write data
   * \param   vas     Functions for accessing metropolis values
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  print_values_with_step_callback(
      const unsigned dstep, const std::initializer_list<value_accessor> &vas,
      const char delim = ' ')
      : _ostr(std::cout), _dstep(dstep), _vas(vas), _delim(delim) 
  {}

  /*! \brief Copy constructor
   *
   * \param   cb      Callback function to copy
   * \return          Callback function
   */
  print_values_with_step_callback(const print_values_with_step_callback &cb)
      : _ostr(cb._ostr), _dstep(cb._dstep), _vas(cb._vas), _delim(cb._delim) {}

  ~print_values_with_step_callback() {}

  void operator()(const metropolis &);

private:
  std::ostream &_ostr;
  unsigned _dstep;
  std::vector<value_accessor> _vas;
  char _delim;
};

/*! \brief Callback for printing data values to a delimited file
 */
template <size_t N> class print_vector_with_step_callback {
public:
  /*! \brief Constructor for callback function that prints data
   *
   * \param   ostr    Output stream
   * \param   dstep   Frequency with which to write data
   * \param   ras     Row accessor
   * \param   delim   Character delimiter
   * \return          Callback function
   */
  print_vector_with_step_callback<N>(
      std::ostream &ostr, const unsigned dstep,
      std::function<std::array<double, N>(const metropolis&)> ras, 
      const char delim = ',')
      : _ostr(ostr), _dstep(dstep), _ras(ras), _delim(delim)
  {}

  ~print_vector_with_step_callback<N>() {}

  void operator()(const metropolis &sim) {
    if (sim.step() % _dstep == 0) {
      _ostr << sim.step();
      const auto &values = _ras(sim);
      for (const auto &value : values) _ostr << _delim << value;
      _ostr << '\n';
    }
  }

private:
  std::ostream &_ostr;
  unsigned _dstep;
  std::function<std::array<double, N>(const metropolis&)> _ras;
  char _delim;
};

/*! \brief Print current metropolis step
 *
 * \param   dstep   Frequency with which to print profiling information
 * \param   msg     Message to print
 * \return          Callback
 */
callback print_step(const unsigned dstep, const char *msg=""); 

/*! \brief Print step required to simulate dstep steps
 *
 * \param   dstep  Frequency with which to print profiling information
 * \return         Callback
 */
callback print_profile(const unsigned dstep);

/*! \brief Outputs xyz data to an output stream in the xyz format
 *
 * \param   ostr        Output stream
 * \param   da          Data accessor for xyz data
 * \param   sim         metropolis object
 * \param   prepend_id  Flag, if true prepend each row with molecular id
 */
void output_xyz(std::ostream &ostr, data_accessor da, const metropolis &sim,
                const bool prepend_id=true);

/*! \brief Saves xyz data to a file in the xyz format
 *
 * \param   fname       Filename
 * \param   da          Data accessor for xyz data
 * \param   sim         metropolis object
 * \param   prepend_id  Flag, if true prepend each row with molecular id
 */
inline void save_xyz(const char *fname, data_accessor da, 
                     const metropolis &sim, const bool prepend_id=true) {
  std::ofstream outfile(fname, std::ios::out);
  output_xyz(outfile, da, sim, prepend_id);
}

const static unsigned _w = 7; //!< to be used in print_trajectory_callback
const static unsigned _p = 3; //!< to be used in print_trajectory_callback

/*! \brief Print molecule trajectories and diagnostic information
 *
 * \param   dstep         Frequency with which to print trajectory information
 * \param   ostr          Output stream
 * \param   print_header  Print header for tabulated trajectory values
 * \return                Callback for printing trajectories
 */
auto print_trajectory_callback(const unsigned dstep = 1, 
                               std::ostream &ostr = std::cout,
                               const bool print_header = true) {
  if (dstep < 1) throw std::invalid_argument(std::string("Frequency with which "
                 "to print trajectory information must be positive. ") + 
                 std::to_string(dstep) + std::string(" is not positive.")); 

  if (print_header)
    ostr << std::setw(3*_w+2) << 'x' << ' ' 
         << std::setw(3*_w+2) << "dx" << ' '
         << std::setw(_w) << "choice" << ' '
         << std::setw(_w) << "dU" << ' '
         << std::setw(_w) << "eps" << ' '
         << std::setw(_w) << "accept" << ' '
         << '\n';
  
  return [dstep, &ostr](const metropolis &sim) -> void {
    if (sim.step() % dstep == 0) {
      size_t i;
      for (i = 0; i < sim.D(); ++i) 
        ostr << std::setw(_w) << std::setprecision(_p) << sim.positions()(i, sim.choice()) << ' ';
      if (i < 3) ostr << std::setw((3 - i)*_w) << ' ';

      for (i = 0; i < sim.D(); ++i) 
        ostr << std::setw(_w) << std::setprecision(_p) << sim.dx()(i) << ' ';
      if (i < 3) ostr << std::setw((3 - i)*_w) << ' ';

      ostr << std::setw(_w) << sim.choice() << ' '
           << std::setw(_w) << std::setprecision(_p) << sim.dU() << ' '
           << std::setw(_w) << std::setprecision(_p) << sim.eps() << ' '
           << std::setw(_w) << sim.accepted() << ' '
           << '\n';
    }
  };
}

} // namespace dstep

#endif
