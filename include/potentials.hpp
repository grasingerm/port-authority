#ifndef __POTENTIALS_HPP__
#define __POTENTIALS_HPP__

#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <armadillo>
#include "pauth_types.hpp"

namespace pauth {

/*! \brief Abstract base class for potentials
 *
 * Defines a public interface for all potentials (e.g. spring, LJ, bond-angle,
 * etc.)
 */
class abstract_potential {
public:
  virtual ~abstract_potential() = 0;

  /*! \brief Calculates total potential energy of the current configuration
   *
   * Calculates the total potential energy of the current configuration given
   * each molecule position. Complexity is N^2.
   * 
   * \param    sim             Metropolis simulation object
   * \return                   Potential energy of the current configuration
   */
  inline double U(const metropolis &sim) const {
    return _U(sim);
  }

  /*! \brief Calculates the change in potential energy from a molecule moving
   *
   * Calculates the change in potential energy due to the motion of a single
   * molecule given by the index i. Note the complexity of this is N, which
   * is much more efficient than calculating the difference using `U`.
   * Requires: 0 <= i < N
   * 
   * \param    molecular_ids   Molecular ids
   * \param    j               Index of the molecule that moved
   * \param    dx              Distance moved
   * \return                   Change in potential energy
   */
  inline double delta_U(const metropolis &sim, const size_t j, 
                        arma::vec& dx) const {
    return _delta_U(sim, j, dx);
  }

private:
  virtual double _U(const metropolis &sim) const = 0;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const = 0;
};

/*! Public interface for a 6-12 Lennard-Jones pairwise potential
 */
class abstract_LJ_potential : public abstract_potential {
public:
  virtual ~abstract_LJ_potential() = 0;

  /*! \brief Get the well depth of the potential (epsilon)
   *
   * \param    id1   Molecular id of the first molecule
   * \param    id2   Molecular id of the second molecule
   * \return         Depth of the potential well
   */
  inline double get_well_depth(const molecular_id id1,
                               const molecular_id id2) const {
    return _get_well_depth(id1, id2);
  }

  /*! \brief Get the distance at which the potential is zero (sigma)
   *
   * \param    id1   Molecular id of the first molecule
   * \param    id2   Molecular id of the second molecule
   * \return         Zero of the potential
   */
  inline double get_rzero(const molecular_id id1,
                          const molecular_id id2) const {
    return _get_rzero(id1, id2);
  }

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const;
  virtual double _get_well_depth(const molecular_id,
                                 const molecular_id) const = 0;
  virtual double _get_rzero(const molecular_id, const molecular_id) const = 0;
};

/*! \brief 6-12 Lennard-Jones pairwise potential
 *
 * Lennard-Jones potential to be used in a simulation where the potential
 * function between every pair of molecules is the same (i.e. to be used in
 * a simulation where all of the molecules are of the same type).
 */
class const_well_params_LJ_potential : public abstract_LJ_potential {
public:
  /*! \brief Constructor for a LJ potential with constant well parameters
   *
   * \param    well_depth    Depth of the potential well
   * \param    rzero         Finite distance at which potential is zero
   * \return                 LJ potential
   */
  const_well_params_LJ_potential(const double well_depth, const double rzero)
      : _well_depth(well_depth), _rzero(rzero) {}

  ~const_well_params_LJ_potential() {}

private:
  virtual double _get_well_depth(molecular_id, molecular_id) const {
    return _well_depth;
  }

  virtual double _get_rzero(molecular_id, molecular_id) const { return _rzero; }

  double _well_depth;
  double _rzero;
};

/*! \brief Public interface for a 6-12 Lennard-Jones potential with a cutoff
 *
 * Public interface for a 6-12 Lennard-Jones potential with a cutoff radius
 * and periodic boundary conditions.
 */
class abstract_LJ_cutoff_potential : public abstract_LJ_potential {
public:
  /*! \brief Constructor for a LJ potential with a cutoff radius
   *
   * \param   edge_length    Length of edge of control volume cube
   * \param   cutoff         Cutoff radius, default value is 2.5
   * \return                 LJ potential
   */
  abstract_LJ_cutoff_potential(const double edge_length,
                               const double cutoff = 2.5)
      : _edge_length(edge_length), _cutoff(cutoff), _rc2(_cutoff * _cutoff),
        _rc6(_rc2 * _rc2 * _rc2), _rc7(_rc6 * _cutoff), _rc12(_rc6 * _rc6),
        _rc13(_rc6 * _rc7) {}

  virtual ~abstract_LJ_cutoff_potential() = 0;

  /*! \brief Get edge length of control volume
   *
   * \return            Edge length
   */
  double get_edge_length() const { return _edge_length; }

  /*! \brief Get cutoff radius
   *
   * \return            Cutoff radius
   */
  double get_cutoff() const { return _cutoff; }

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const;
  virtual double _get_well_depth(const molecular_id,
                                 const molecular_id) const = 0;
  virtual double _get_rzero(const molecular_id, const molecular_id) const = 0;

  double _edge_length;
  double _cutoff;

  double _rc2;
  double _rc6;
  double _rc7;
  double _rc12;
  double _rc13;
};

/*! \brief 6-12 Lennard-Jones pairwise potential with a cutoff radius
 *
 * Lennard-Jones potential to be used in a simulation where the potential
 * function between every pair of molecules is the same (i.e. to be used in
 * a simulation where all of the molecules are of the same type).
 */
class const_well_params_LJ_cutoff_potential
    : public abstract_LJ_cutoff_potential {
public:
  /*! \brief Constructor for a LJ potential with constant well parameters
   *
   * \param    well_depth    Depth of the potential well
   * \param    rzero         Finite distance at which potential is zero
   * \return                 LJ potential
   */
  const_well_params_LJ_cutoff_potential(const double well_depth,
                                        const double rzero,
                                        const double edge_length,
                                        const double cutoff = 2.5)
      : abstract_LJ_cutoff_potential(edge_length, cutoff),
        _well_depth(well_depth), _rzero(rzero) {}

  ~const_well_params_LJ_cutoff_potential() {}

private:
  virtual double _get_well_depth(molecular_id, molecular_id) const {
    return _well_depth;
  }

  virtual double _get_rzero(molecular_id, molecular_id) const { return _rzero; }

  double _well_depth;
  double _rzero;
};

/*! \brief Public interface for a spring potential
 */
class abstract_spring_potential : public abstract_potential {
public:
  virtual ~abstract_spring_potential() = 0;

  /*! \brief Get the spring constant
   *
   * \param    id    Molecular id of the molecule
   * \return         Spring constant
   */
  inline double get_k(molecular_id id) const { return _get_k(id); }

private:
  virtual double _U(const metropolis &sim) const;
  virtual double  _delta_U(const metropolis &sim, const size_t j, 
                           arma::vec &dx) const;

  virtual double _get_k(molecular_id) const = 0;
};

/*! \brief Potential due to a spring with constant k
 *
 * Potential due to a spring in which the spring constant, k, is the same
 * for all molecules, regardless of molecular id.
 * k should always be greater than 0.
 */
class const_k_spring_potential : public abstract_spring_potential {
public:
  /*! \brief Constructor for potential with same spring constant for all
   * molecules
   *
   * Constructor for potential with same spring constant for all molecules.
   * Requires: k > 0
   *
   * \param    k     Spring constant
   * \return         Spring potential
   */
  const_k_spring_potential(const double k) : _k(k) {}

  virtual ~const_k_spring_potential() {}

private:
  virtual double _get_k(molecular_id) const { return _k; }

  double _k;
};

/*! \brief Potential due to a quadratic spring with constant parameters
 *
 * Potential due to a spring in which the spring parameters are the same
 * for all molecules, regardless of molecular id.
 */
class const_quad_spring_potential : public abstract_potential {
public:
  /*! \brief Constructor for potential with same spring parameters for all
   * molecules
   *
   * Constructor for potential with same spring constant for all molecules.
   *
   * \param    k     Spring constant
   * \return         Spring potential
   */
  const_quad_spring_potential(const double a, const double b, const double c)
      : a(a), b(b), c(c) {}

  virtual ~const_quad_spring_potential() {}

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const;

  double a;
  double b;
  double c;
};

/*! \brief Potential due to a spring with a polynomial potential
 *
 * Potential due to a spring in which the potential, a x^n + b x^-1 + ...,
 * is the same for all molecules, regardless of molecular id.
 */
class const_poly_spring_potential : public abstract_potential {
public:
  /*! \brief Constructor for potential with same spring potential for all
   * molecules
   *
   * Constructor for potential with same spring constant for all molecules.
   *
   * \param    coeffs   Polynomial coefficients
   * \return            Spring potential
   */
  const_poly_spring_potential(const std::initializer_list<double> &coeffs);

  virtual ~const_poly_spring_potential() {}

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const;

  std::vector<double> pcoeffs;
  std::vector<double> fcoeffs;
};

} // namespace pauth

#endif
