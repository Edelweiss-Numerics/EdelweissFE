/* ---------------------------------------------------------------------
 *
 *  _____    _      _              _         _____ _____
 * | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
 * |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
 * | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
 * |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
 *
 *
 *  Unit of Strength of Materials and Structural Analysis
 *  University of Innsbruck,
 *  2017 - today
 *
 *  Alexander Dummer alexander.dummer@uibk.ac.at
 *
 *  This file is part of EdelweissFE.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  The full text of the license can be found in the file LICENSE.md at
 *  the top level directory of EdelweissFE.
 * ---------------------------------------------------------------------
 *
 * A thin C++ shim around Marmot's MarmotMaterialGradientPlasticityHypoElastic, for the very
 * same three reasons as _gradientenhancedshim.h: the base class is templated on a non-type
 * parameter which Cython cannot express, the interface exchanges fixed-size Eigen objects
 * which are marshalled here in a well defined row-major layout, and Marmot stores only the
 * pointer to the material properties, of which the shim therefore keeps its own copy.
 */

#pragma once

#include "Marmot/MarmotMaterialGradientPlasticityHypoElastic.h"
#include "Marmot/MarmotMaterialGradientPlasticityHypoElasticFactory.h"
#include "Marmot/MarmotTypedefs.h"
#include "Marmot/MarmotUtils.h"

#include <Eigen/Core>
#include <stdexcept>
#include <string>
#include <vector>

namespace EdelweissFE {

  /**
   * @brief Non-templated facade for MarmotMaterialGradientPlasticityHypoElastic< nYieldSurfaces >.
   * @tparam nYieldSurfaces The number of yield surfaces.
   */
  template < int nYieldSurfaces >
  class GradientPlasticityHypoElasticShim {

    using Material = MarmotMaterialGradientPlasticityHypoElastic< nYieldSurfaces >;

    /// Eigen rejects RowMajor storage for matrices with a single column, and for a single
    /// column the two storage orders coincide anyway.
    static constexpr int storageOrder( int nColumns ) { return nColumns == 1 ? Eigen::ColMajor : Eigen::RowMajor; }

    using YieldVector = Eigen::Matrix< double, nYieldSurfaces, 1 >;
    using StressYield = Eigen::Matrix< double, 6, nYieldSurfaces, storageOrder( nYieldSurfaces ) >;
    using YieldStress = Eigen::Matrix< double, nYieldSurfaces, 6, Eigen::RowMajor >;
    using YieldMatrix = Eigen::Matrix< double, nYieldSurfaces, nYieldSurfaces, storageOrder( nYieldSurfaces ) >;
    using Stiffness   = Eigen::Matrix< double, 6, 6, Eigen::RowMajor >;

    std::vector< double > ownedMaterialProperties;
    Material*             material = nullptr;

  public:
    GradientPlasticityHypoElasticShim( const std::string& materialName,
                                       const double*      materialProperties,
                                       int                nMaterialProperties,
                                       int                materialNumber )
      : ownedMaterialProperties( materialProperties, materialProperties + nMaterialProperties )
    {
      material = MarmotLibrary::MarmotMaterialGradientPlasticityHypoElasticFactory<
        nYieldSurfaces >::createMaterial( materialName,
                                          ownedMaterialProperties.data(),
                                          static_cast< int >( ownedMaterialProperties.size() ),
                                          materialNumber );

      if ( material == nullptr )
        throw std::invalid_argument( "Marmot does not provide a gradient plasticity hypoelastic material '" +
                                     materialName + "'" );
    }

    ~GradientPlasticityHypoElasticShim() { delete material; }

    GradientPlasticityHypoElasticShim( const GradientPlasticityHypoElasticShim& )            = delete;
    GradientPlasticityHypoElasticShim& operator=( const GradientPlasticityHypoElasticShim& ) = delete;

    static int getNumberOfYieldSurfaces() { return nYieldSurfaces; }

    int getNumberOfRequiredStateVars() const { return material->getNumberOfRequiredStateVars(); }

    void initializeYourself( double* stateVars, int nStateVars )
    {
      material->initializeYourself( stateVars, nStateVars );
    }

    double getDensity( const double* stateVars ) const { return material->getDensity( stateVars ); }

    StateView getStateView( const std::string& stateName, double* stateVars ) const
    {
      return material->getStateView( stateName, stateVars );
    }

    /**
     * @brief Evaluate the material.
     *
     * All matrix arguments use row-major storage. @p stress carries the stress at the
     * beginning of the increment on input and the updated stress on output, mirroring
     * Marmot's response struct.
     *
     * @param planeStress If true, Marmot's plane stress algorithm is used, i.e. the
     *                    out-of-plane strain is condensed out by the material itself.
     */
    void computeStress( double*       stress,
                        double*       f,
                        double*       elasticEnergyDensity,
                        double*       dissipation,
                        double*       dStress_dStrain,
                        double*       dStress_dLambda,
                        double*       dStress_dLaplacian,
                        double*       dF_dStrain,
                        double*       dF_dLambda,
                        double*       dF_dLaplacian,
                        const double* dStrain,
                        const double* dLambda,
                        const double* laplaceDLambda,
                        double*       stateVars,
                        double        time,
                        double        dT,
                        bool          planeStress ) const
    {
      // Named maps rather than temporaries: `Eigen::Map< T >( ptr ) = value;` would be
      // parsed as a declaration of a variable named `ptr`, not as an assignment.
      Eigen::Map< Marmot::Vector6d > stressMap( stress );
      Eigen::Map< YieldVector >      fMap( f );

      Eigen::Map< Stiffness >   dStress_dStrainMap( dStress_dStrain );
      Eigen::Map< StressYield > dStress_dLambdaMap( dStress_dLambda );
      Eigen::Map< StressYield > dStress_dLaplacianMap( dStress_dLaplacian );
      Eigen::Map< YieldStress > dF_dStrainMap( dF_dStrain );
      Eigen::Map< YieldMatrix > dF_dLambdaMap( dF_dLambda );
      Eigen::Map< YieldMatrix > dF_dLaplacianMap( dF_dLaplacian );

      typename Material::increment inc;
      inc.dStrain        = Eigen::Map< const Marmot::Vector6d >( dStrain );
      inc.dLambda        = Eigen::Map< const YieldVector >( dLambda );
      inc.laplaceDLambda = Eigen::Map< const YieldVector >( laplaceDLambda );
      inc.time           = time;
      inc.dT             = dT;

      typename Material::response res;
      res.stress               = stressMap;
      res.f                    = YieldVector::Zero();
      res.stateVars            = stateVars;
      res.elasticEnergyDensity = 0.0;
      res.dissipation          = 0.0;

      typename Material::tangents tan;

      if ( planeStress )
        material->computePlaneStress( res, tan, inc );
      else
        material->computeStress( res, tan, inc );

      stressMap = res.stress;
      fMap      = res.f;

      *elasticEnergyDensity = res.elasticEnergyDensity;
      *dissipation          = res.dissipation;

      dStress_dStrainMap    = tan.dStressddStrain;
      dStress_dLambdaMap    = tan.dStressddLambda;
      dStress_dLaplacianMap = tan.dStressddLaplacian;
      dF_dStrainMap         = tan.dFddStrain;
      dF_dLambdaMap         = tan.dFddLambda;
      dF_dLaplacianMap      = tan.dFddLaplacian;
    }
  };

  /// The instantiation used for materials with a single yield surface, e.g. GradientVonMises
  /// and GradientLinearElastic.
  using GradientPlasticityHypoElasticShim1 = GradientPlasticityHypoElasticShim< 1 >;

} // namespace EdelweissFE
