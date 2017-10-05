#ifndef __POTENTIALS_HPP__
#define __POTENTIALS_HPP__

#include <initializer_list>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include <map>
#include <functional>
#include <armadillo>
#include <stdexcept>
#include <boost/range/combine.hpp>
#include "pauth_types.hpp"
#include "molecular.hpp"

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
   * \param    sim             Metropolis simulation object
   * \param    j               Index of the molecule that moved
   * \param    rn_j            New position
   * \return                   Change in potential energy
   */
  inline double delta_U(const metropolis &sim, const size_t j, 
                        arma::vec& rn_j) const {
    return _delta_U(sim, j, rn_j);
  }

  /*! \brief Calculates the force between molecules i and j
   *
   * \param    sim             Metropolis simulation object
   * \param    i               Index of the first molecule
   * \param    j               Index of the second molecule
   * \return                   Force of molecule j on molecule i
   */
  inline arma::vec forceij(const metropolis &sim, const size_t i, 
                           const size_t j) const {
    return _forceij(sim, i, j);
  }

private:
  virtual double _U(const metropolis &sim) const = 0;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &rn_j) const = 0;
  virtual arma::vec _forceij(const metropolis &sim, const size_t i, 
                             const size_t j) const = 0;
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
  virtual double _U(const metropolis &sim) const = 0;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const = 0;
  virtual arma::vec _forceij(const metropolis &sim, const size_t i, 
                             const size_t j) const = 0;
  virtual double _get_well_depth(const molecular_id,
                                 const molecular_id) const = 0;
  virtual double _get_rzero(const molecular_id, const molecular_id) const = 0;
};

/*! Public interface for a 6-12 Lennard-Jones pairwise potential without cutoff
 */
class abstract_LJ_full_potential : virtual public abstract_LJ_potential {
public:
  virtual ~abstract_LJ_full_potential() = 0;

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const;
  virtual arma::vec _forceij(const metropolis &sim, const size_t i, 
                             const size_t j) const;
};

/*! \brief 6-12 Lennard-Jones pairwise potential
 *
 * Public interface for a Lennard-Jones potential to be used in a simulation 
 * where the potential parameters between every pair of molecules is loaded 
 * from a YAML file and looked up at runtime.
 */
class abstract_LJ_lookup_potential : virtual public abstract_LJ_potential {
public:
  virtual ~abstract_LJ_lookup_potential() = 0;

  /*! \brief Constructor for LJ potential where well parameters are looked up
   *
   * \param       fname     Filename to well parameter data
   * \return                LJ potential with well parameter lookup
   */
  abstract_LJ_lookup_potential(const char *fname);

private:
  virtual double _get_well_depth(const molecular_id, const molecular_id) const;
  virtual double _get_rzero(const molecular_id, const molecular_id) const;

  molecular_pair_interaction_map _well_depth_map; 
  molecular_pair_interaction_map _rzero_map; 
};

/*! \brief 6-12 Lennard-Jones pairwise potential
 *
 * Lennard-Jones potential to be used in a simulation where the potential
 * parameters between every pair of molecules is loaded from a YAML file
 * and looked up at runtime.
 */
class LJ_potential : public abstract_LJ_full_potential, 
                     public abstract_LJ_lookup_potential {
public:
  /*! \brief Constructor for LJ potential where well parameters are looked up
   *
   * \param       fname     Filename to well parameter data
   * \return                LJ potential with well parameter lookup
   */
  LJ_potential(const char *fname) 
    : abstract_LJ_lookup_potential(fname) {}
};

/*! \brief 6-12 Lennard-Jones pairwise potential
 *
 * Lennard-Jones potential to be used in a simulation where the potential
 * function between every pair of molecules is the same (i.e. to be used in
 * a simulation where all of the molecules are of the same type).
 */
class const_well_params_LJ_potential : public abstract_LJ_full_potential {
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
class abstract_LJ_cutoff_potential : virtual public abstract_LJ_potential {
public:
  /*! \brief Constructor for a LJ potential with a cutoff radius
   *
   * \param   cutoff         Cutoff radius, default value is 2.5
   * \return                 LJ potential
   */
  abstract_LJ_cutoff_potential(const double cutoff = 2.5)
      : _cutoff(cutoff), _rc2(_cutoff * _cutoff),
        _rc6(_rc2 * _rc2 * _rc2), _rc7(_rc6 * _cutoff), _rc12(_rc6 * _rc6),
        _rc13(_rc6 * _rc7) {}

  virtual ~abstract_LJ_cutoff_potential() = 0;

  /*! \brief Get cutoff radius
   *
   * \return            Cutoff radius
   */
  inline double cutoff() const { return _cutoff; }

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &rn_j) const;
  virtual arma::vec _forceij(const metropolis &sim, const size_t i, 
                             const size_t j) const;

  double _cutoff;

  double _rc2;
  double _rc6;
  double _rc7;
  double _rc12;
  double _rc13;
};

/*! \brief 6-12 Lennard-Jones pairwise potential
 *
 * Lennard-Jones potential with cutoff to be used in a simulation where the 
 * parameters between every pair of molecules is loaded from a YAML file
 * and looked up at runtime.
 */
class LJ_cutoff_potential : public abstract_LJ_lookup_potential, 
                            public abstract_LJ_cutoff_potential {
public:
  /*! \brief Constructor for LJ potential where well parameters are looked up
   *
   * \param       fname     Filename to well parameter data
   * \param       cutoff    Cutoff radius
   * \return                LJ potential with well parameter lookup
   */
  LJ_cutoff_potential(const char *fname, const double cutoff = 2.5) 
    : abstract_LJ_lookup_potential(fname), abstract_LJ_cutoff_potential(cutoff) 
      {}
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
                                        const double cutoff = 2.5)
      : abstract_LJ_cutoff_potential(cutoff),
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
                           arma::vec &rn_j) const;
  virtual arma::vec _forceij(const metropolis &sim, const size_t, 
                             const size_t) const;

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
                          arma::vec &rn_j) const;
  virtual arma::vec _forceij(const metropolis &sim, const size_t, 
                             const size_t) const;

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
                          arma::vec &rn_j) const;
  virtual arma::vec _forceij(const metropolis &sim, const size_t, 
                             const size_t) const;

  std::vector<double> pcoeffs;
  std::vector<double> fcoeffs;
};

/*! \brief Two state potential with interactions between molecules
 */
class twostate_int_potential : public abstract_potential {
public:
  /*! Constructor for two state potential with interactions
   *
   * \param     gamma     Potential energy of state 1
   * \param     mu        Potential energy of state 2
   * \return              Two state potential with interactions
   */
  twostate_int_potential(const double gamma, const double mu) {
    _Us[0] = gamma;
    _Us[1] = mu;
  }

private:
  std::array<double, 2> _Us;

  double _U(const metropolis &sim) const;
  double _delta_U(const metropolis &sim, const size_t j, 
                  arma::vec &rn_j) const;
  inline double _Ui(const arma::vec &x) const {
    return _Us[static_cast<unsigned>(x(0))];
  }
  inline bool _check_x(const arma::vec &x) const {
    return (static_cast<unsigned>(x(0)) == 0 ||
            static_cast<unsigned>(x(0)) == 1);
  }
  arma::vec _forceij(const metropolis &sim, const size_t, const size_t) const;
};


/*! \brief Potential of a dipole in an electric field
 */
class abstract_dipole_electric_potential : public abstract_potential {
public:
  virtual ~abstract_dipole_electric_potential() = 0;

  /*! \brief Get electric field
   *
   * \param    x     X coordinates where to find electric field
   * \return         Electric field at x
   */
  inline arma::vec E0(const arma::vec& x) const {
    return _E0(x);
  }

  /*! \brief Get the molecular dipole
   *
   * \param    id    Molecular id
   * \param    x     X coordinates of the dipole
   * \return         Susceptibility tensor
   */
  inline arma::vec dipole(const molecular_id id, const arma::vec& x) const {
    return _dipole(id, x);
  }

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const;
  virtual arma::vec _forceij(const metropolis &sim, const size_t i, 
                             const size_t j) const;
  virtual arma::vec _E0(const arma::vec& x) const = 0;
  virtual arma::vec _dipole(const molecular_id id, const arma::vec& x) 
    const = 0;
};

/*! \brief Potential of a dipole in a constant electric field
 */
class const_E_const_suscept_dipole_electric_potential 
  : public abstract_dipole_electric_potential {

public:
  virtual ~const_E_const_suscept_dipole_electric_potential() {}
 
  /*! \brief Dipole in a constant electric field with a constant susceptbility tensor
   *
   * \param     E0          Electric field
   * \param     dipole_f    Dipole function
   * \return                Dipole potential
   */
  const_E_const_suscept_dipole_electric_potential(arma::vec E0,
      std::function<arma::vec(const arma::vec&, const arma::vec&)> dipole_f) :
      _const_E0(E0), _dipole_f(dipole_f) {}

private:
  virtual arma::vec _E0(const arma::vec&) const {
    return _const_E0;
  }
  virtual arma::vec _dipole(const molecular_id, const arma::vec& x) const {
    return _dipole_f(x, _const_E0);
  }

  arma::vec _const_E0;
  std::function<arma::vec(const arma::vec& x, const arma::vec& E0)> _dipole_f;

};

/*! \brief Dipole potential for a constant electric field, param by two angles
 *
 * \param       E0          Electric field
 * \param       mu          Fixed dipole magnitude
 * \return                  Dipole potential
 */
const_E_const_suscept_dipole_electric_potential 
  const_E_fixed_2D_dipole_potential(arma::vec E0, const double mu);

/*! \brief Dipole potential for a constant electric field, param by two angles
 *
 * \param       E0          Electric field
 * \param       kappa       Susceptibility parameter
 * \return                  Dipole potential
 */
const_E_const_suscept_dipole_electric_potential 
  const_E_uniaxial_2D_dipole_potential(arma::vec E0, const double kappa);

/*! \brief Dipole potential for a constant electric field, param by two angles
 *
 * \param       E0          Electric field
 * \param       kappa       Susceptibility parameter
 * \return                  Dipole potential
 */
const_E_const_suscept_dipole_electric_potential 
  const_E_TI_2D_dipole_potential(arma::vec E0, const double kappa);

} // namespace pauth

#endif
