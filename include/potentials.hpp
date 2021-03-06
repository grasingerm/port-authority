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
class dipole_electric_potential : public abstract_potential {
public:
  /*! \brief Potential of a dipole in a (constant) electric field
   *
   * \param   E0        Magnitude of electric field
   * \param   dof_idx   Index of the degree of freedom of the E field
   * \return            Potential of a dipole in a (constant) electric field
   */
  dipole_electric_potential(const double E0, const unsigned dof_idx = 2) {
    _efield[dof_idx] = [E0](const arma::vec&) { return E0; };
  }

  /*! \brief Potential of a dipole in a (constant) electric field
   *
   * \param   E0        Magnitude of electric field
   * \param   dof_idx   Index of the degree of freedom of the E field
   * \return            Potential of a dipole in a (constant) electric field
   */
  dipole_electric_potential(const std::initializer_list<double> Es, 
                            const std::initializer_list<unsigned> dof_idxs) {
    if (Es.size() != dof_idxs.size()) 
      throw std::invalid_argument("E field function list and dof list should "
                                  "be of the same length");

    for(auto tup : boost::combine(dof_idxs, Es)) {
      unsigned idx;
      double E;
      boost::tie(idx, E) = tup;
      _efield[idx] = [E](const arma::vec&) { return E; };
    }
  }

  /*! \brief Potential of a dipole in an electric field
   *
   * \param   Es        Components of electric field
   * \param   dox_idxs  Indexes of the degree of freedom of the E field
   * \return            Potential of a dipole in an electric field
   */
  dipole_electric_potential(std::initializer_list<
      std::function<double(const arma::vec&)> > Es, 
      std::initializer_list<unsigned> dof_idxs) {

    if (Es.size() != dof_idxs.size()) 
      throw std::invalid_argument("E field function list and dof list should "
                                  "be of the same length");

    for(auto tup : boost::combine(dof_idxs, Es)) 
      _efield[boost::get<0>(tup)] = boost::get<1>(tup); 

  }

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const;
  virtual arma::vec _forceij(const metropolis &sim, const size_t i, 
                             const size_t j) const;

  std::map<unsigned, std::function<double(const arma::vec&)>> _efield;
};

/*! \brief Potential of a dipole in an electric field
 */
class abstract_dipole_strain_potential : public abstract_potential {
public:
  virtual ~abstract_dipole_strain_potential() = 0;

  /*! \brief Inverse of the susceptibility tensor
   *
   * \param   xs    Degrees of freedom of dipole
   * \param   id    Molecular id of the particle
   * \return        Inverse susceptibility
   */
  inline arma::mat inv_chi(const arma::vec& xs, const molecular_id id) const {
    return _inv_chi(xs, id);
  }

  /*! \brief Polarization vector
   *
   * \param   xs    Degrees of freedom of dipole
   * \return        Polarization vector
   */
  inline arma::vec p(const arma::vec& xs) const {
    return _p(xs);
  }

private:
  virtual double _U(const metropolis &sim) const;
  virtual double _delta_U(const metropolis &sim, const size_t j, 
                          arma::vec &dx) const;
  virtual arma::vec _forceij(const metropolis &sim, const size_t, 
                             const size_t) const;
  virtual arma::mat _inv_chi(const arma::vec& xs, 
                             const molecular_id id) const = 0;
  virtual arma::vec _p(const arma::vec& xs) const = 0;
};

/*! \brief Dipole in 2D space
 */
class abstract_dipole_strain_2d_potential : 
  virtual public abstract_dipole_strain_potential {
public:
  virtual ~abstract_dipole_strain_2d_potential()=0;
private:
  arma::vec _p(const arma::vec& xs) const {
    return arma::vec(xs.memptr(), 2);
  }
};

/*! \brief Dipole in 3D space
 */
class abstract_dipole_strain_3d_potential :
  virtual public abstract_dipole_strain_potential {
public:
  virtual ~abstract_dipole_strain_3d_potential()=0;
private:
  arma::vec _p(const arma::vec& xs) const {
    return arma::vec(xs.memptr(), 3);
  }
};

/*! \brief Dipole whose susceptibility does not depend on space, direction, etc.
 */
class abstract_dipole_strain_linear_potential :
  virtual public abstract_dipole_strain_potential {
public:
  virtual ~abstract_dipole_strain_linear_potential()=0;

  abstract_dipole_strain_linear_potential(const arma::mat& inv_chi) 
    : _const_inv_chi(inv_chi) {}

private:
  arma::mat _inv_chi(const arma::vec&, const molecular_id) const {
    return _const_inv_chi;
  }

  arma::mat _const_inv_chi;
};

/*! \brief Dipole, in 2D space, whose susceptibility tensor is linear
 */
class dipole_strain_linear_2d_potential :
  public abstract_dipole_strain_2d_potential,
  public abstract_dipole_strain_linear_potential {
public:
  virtual ~dipole_strain_linear_2d_potential() {}

  dipole_strain_linear_2d_potential(const arma::mat& inv_chi)
    : abstract_dipole_strain_linear_potential(inv_chi) {}
};

/*! \brief Dipole, in 3D space, whose susceptibility tensor is linear
 */
class dipole_strain_linear_3d_potential :
  public abstract_dipole_strain_3d_potential,
  public abstract_dipole_strain_linear_potential {
public:
  virtual ~dipole_strain_linear_3d_potential() {}

  dipole_strain_linear_3d_potential(const arma::mat& inv_chi)
    : abstract_dipole_strain_linear_potential(inv_chi) {}
};

/*! \brief Dipole, in 2D space, whose susceptibility tensor is nonlinear
 */
class dipole_strain_nonlinear_2d_potential : 
  public abstract_dipole_strain_2d_potential {
public:
  virtual ~dipole_strain_nonlinear_2d_potential() {}

  dipole_strain_nonlinear_2d_potential(std::function<arma::mat(const arma::vec&,
    const molecular_id)> inv_chi_f) : _inv_chi_f(inv_chi_f) {}
private:
  arma::mat _inv_chi(const arma::vec& xs, const molecular_id id) const {
    const double theta = xs(2);
    arma::vec n({std::cos(theta), std::sin(theta)});
    return _inv_chi_f(n, id);
  }

  std::function<arma::mat(const arma::vec&, const molecular_id)> _inv_chi_f;
};

/*! \brief Dipole, in 3D space, whose susceptibility tensor is nonlinear
 */
class dipole_strain_nonlinear_3d_potential : 
  public abstract_dipole_strain_3d_potential {
public:
  virtual ~dipole_strain_nonlinear_3d_potential() {}

  dipole_strain_nonlinear_3d_potential(std::function<arma::mat(const arma::vec&,
    const molecular_id)> inv_chi_f) : _inv_chi_f(inv_chi_f) {}
private:
  arma::mat _inv_chi(const arma::vec& xs, const molecular_id id) const {
    const double phi = xs(3);
    const double theta = xs(4);
    const double cp = std::cos(phi);
    const double sp = std::sin(phi);
    const double ct = std::cos(theta);
    const double st = std::sin(theta);
    arma::vec n({cp * st, sp * st, ct});
    return _inv_chi_f(n, id);
  }

  std::function<arma::mat(const arma::vec&, const molecular_id)> _inv_chi_f;
};

} // namespace pauth

#endif
