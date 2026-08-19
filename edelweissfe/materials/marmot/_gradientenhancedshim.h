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
 * A thin C++ shim around Marmot's MarmotMaterialGeneralGradientEnhancedHypoElastic.
 *
 * It exists for three reasons:
 *
 *  1. The Marmot base class is templated on the *number* of nonlocal variables,
 *     i.e., on a non-type template parameter. Cython can only express type template
 *     parameters, so the instantiation has to happen in C++.
 *  2. The Marmot interface exchanges fixed-size Eigen objects. Marshalling them here,
 *     to and from plain double pointers in a well defined row-major layout, keeps the
 *     Cython layer free of Eigen and free of storage-order ambiguity.
 *  3. Marmot materials store the *pointer* to their material properties without taking
 *     ownership. The shim keeps its own copy, so the lifetime of the properties can
 *     never depend on what the Python caller does with the original array.
 */

#pragma once

#include "Marmot/MarmotMaterialGeneralGradientEnhancedHypoElastic.h"
#include "Marmot/MarmotMaterialGeneralGradientEnhancedHypoElasticFactory.h"
#include "Marmot/MarmotTypedefs.h"
#include "Marmot/MarmotUtils.h"

#include <Eigen/Core>
#include <stdexcept>
#include <string>
#include <vector>

namespace EdelweissFE {

  /**
   * @brief Non-templated facade for MarmotMaterialGeneralGradientEnhancedHypoElastic< nNonlocal >.
   * @tparam nNonlocal The number of nonlocal variables.
   */
  template < int nNonlocal >
  class GradientEnhancedHypoElasticShim {

    using Material = MarmotMaterialGeneralGradientEnhancedHypoElastic< nNonlocal >;

    /// Eigen rejects RowMajor storage for matrices with a single column, and for a single
    /// column the two storage orders coincide anyway.
    static constexpr int storageOrder( int nColumns ) { return nColumns == 1 ? Eigen::ColMajor : Eigen::RowMajor; }

    using NonlocalVector = Eigen::Matrix< double, nNonlocal, 1 >;
    using StressNonlocal = Eigen::Matrix< double, 6, nNonlocal, storageOrder( nNonlocal ) >;
    using NonlocalStress = Eigen::Matrix< double, nNonlocal, 6, Eigen::RowMajor >;
    using NonlocalMatrix = Eigen::Matrix< double, nNonlocal, nNonlocal, storageOrder( nNonlocal ) >;
    using Stiffness      = Eigen::Matrix< double, 6, 6, Eigen::RowMajor >;

    std::vector< double > ownedMaterialProperties;
    Material*             material = nullptr;

  public:
    GradientEnhancedHypoElasticShim( const std::string& materialName,
                                     const double*      materialProperties,
                                     int                nMaterialProperties,
                                     int                materialNumber )
      : ownedMaterialProperties( materialProperties, materialProperties + nMaterialProperties )
    {
      material = MarmotLibrary::MarmotMaterialGeneralGradientEnhancedHypoElasticFactory<
        nNonlocal >::createMaterial( materialName,
                                     ownedMaterialProperties.data(),
                                     static_cast< int >( ownedMaterialProperties.size() ),
                                     materialNumber );

      if ( material == nullptr )
        throw std::invalid_argument( "Marmot does not provide a gradient enhanced hypoelastic material '" +
                                     materialName + "'" );
    }

    ~GradientEnhancedHypoElasticShim() { delete material; }

    GradientEnhancedHypoElasticShim( const GradientEnhancedHypoElasticShim& )            = delete;
    GradientEnhancedHypoElasticShim& operator=( const GradientEnhancedHypoElasticShim& ) = delete;

    static int getNumberOfNonlocalVariables() { return nNonlocal; }

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
     * @brief Fill @p viscosity with the nonlocal viscosities.
     * @param viscosity Buffer of at least nNonlocal doubles.
     */
    void getNonlocalViscosity( const double* stateVars, double* viscosity ) const
    {
      const auto values = material->getNonlocalViscosity( stateVars );
      for ( int i = 0; i < nNonlocal; ++i )
        viscosity[i] = i < static_cast< int >( values.size() ) ? values[i] : 0.0;
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
                        double*       KLocal,
                        double*       c,
                        double*       elasticEnergyDensity,
                        double*       dissipation,
                        double*       dStress_dStrain,
                        double*       dStress_dK,
                        double*       dKLocal_dStrain,
                        double*       dKLocal_dK,
                        double*       dc_dK,
                        double*       d2c_dK2,
                        const double* dStrain,
                        const double* K,
                        const double* dK,
                        double*       stateVars,
                        double        time,
                        double        dT,
                        bool          planeStress ) const
    {
      // Named maps rather than temporaries: `Eigen::Map< T >( ptr ) = value;` would be
      // parsed as a declaration of a variable named `ptr`, not as an assignment.
      Eigen::Map< Marmot::Vector6d > stressMap( stress );
      Eigen::Map< NonlocalVector >   KLocalMap( KLocal );
      Eigen::Map< NonlocalVector >   cMap( c );

      Eigen::Map< Stiffness >      dStress_dStrainMap( dStress_dStrain );
      Eigen::Map< StressNonlocal > dStress_dKMap( dStress_dK );
      Eigen::Map< NonlocalStress > dKLocal_dStrainMap( dKLocal_dStrain );
      Eigen::Map< NonlocalMatrix > dKLocal_dKMap( dKLocal_dK );
      Eigen::Map< NonlocalMatrix > dc_dKMap( dc_dK );
      Eigen::Map< NonlocalMatrix > d2c_dK2Map( d2c_dK2 );

      typename Material::increment inc;
      inc.dStrain = Eigen::Map< const Marmot::Vector6d >( dStrain );
      inc.K       = Eigen::Map< const NonlocalVector >( K );
      inc.dK      = Eigen::Map< const NonlocalVector >( dK );
      inc.time    = time;
      inc.dT      = dT;

      typename Material::response res;
      res.stress               = stressMap;
      res.KLocal               = KLocalMap;
      res.c                    = NonlocalVector::Zero();
      res.stateVars            = stateVars;
      res.elasticEnergyDensity = 0.0;
      res.dissipation          = 0.0;

      typename Material::tangents tan;

      if ( planeStress )
        material->computePlaneStress( res, tan, inc );
      else
        material->computeStress( res, tan, inc );

      stressMap = res.stress;
      KLocalMap = res.KLocal;
      cMap      = res.c;

      *elasticEnergyDensity = res.elasticEnergyDensity;
      *dissipation          = res.dissipation;

      dStress_dStrainMap = tan.dStressddStrain;
      dStress_dKMap      = tan.dStressddK;
      dKLocal_dStrainMap = tan.dKLocalddStrain;
      dKLocal_dKMap      = tan.dKLocalddK;
      dc_dKMap           = tan.dcddK;
      d2c_dK2Map         = tan.d2cddK2;
    }
  };

  /// The instantiation used for materials with a single nonlocal variable, e.g., GCDP,
  /// LGCDP, PCDP and AT2PhaseField.
  using GradientEnhancedHypoElasticShim1 = GradientEnhancedHypoElasticShim< 1 >;

} // namespace EdelweissFE
