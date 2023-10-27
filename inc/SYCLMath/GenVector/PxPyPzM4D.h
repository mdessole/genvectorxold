// @(#)root/mathcore:$Id: 464c29f33a8bbd8462a3e15b7e4c30c6f5b74a30 $
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT MathLib Team                         *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class PxPyPzM4D
//
// Created by: fischler at Wed Jul 20   2005
//   (starting from PxPyPzM4D by moneta)
//
// Last update: $Id: 464c29f33a8bbd8462a3e15b7e4c30c6f5b74a30 $
//
#ifndef ROOT_Experimental_GenVector_PxPyPzM4D
#define ROOT_Experimental_GenVector_PxPyPzM4D  1

#undef __MAKECINT__


#include "SYCLMath/GenVector/MathUtil.h"


namespace ROOT {

namespace Experimental {

//__________________________________________________________________________________________
/**
    Class describing a 4D coordinate system
    or momentum-energy vectors stored as (Px, Py, Pz, M).
    This system is useful to describe ultra-relativistic particles
    (like electrons at LHC) to avoid numerical errors evaluating the mass
    when E >>> m
    The metric used is (-,-,-,+)
    Spacelike particles (M2 < 0) are described with negative mass values,
    but in this case m2 must alwasy be less than P2 to preserve a positive value of E2

    @ingroup GenVector

    @sa Overview of the @ref GenVector "physics vector library"
*/

template <class ScalarType = double>
class PxPyPzM4D {

public :

   typedef ScalarType Scalar;

   // --------- Constructors ---------------

   /**
      Default constructor  with x=y=z=m=0
   */
   __roohost__ __roodevice__  PxPyPzM4D() : fX(0.0), fY(0.0), fZ(0.0), fM(0.0) { }


   /**
      Constructor  from x, y , z , m values
   */
   __roohost__ __roodevice__  PxPyPzM4D(Scalar px, Scalar py, Scalar pz, Scalar m) :
      fX(px), fY(py), fZ(pz), fM(m) {

      if (fM < 0) RestrictNegMass();
   }

   /**
      construct from any 4D  coordinate system class
      implementing X(), Y(), X() and M()
   */
   template <class CoordSystem>
   __roohost__ __roodevice__  explicit PxPyPzM4D(const CoordSystem & v) :
      fX( v.X() ), fY( v.Y() ), fZ( v.Z() ), fM( v.M() )
   { }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // so we decided to re-implement them ( there is no no need to have them with g++4)
   /**
      copy constructor
    */
   __roohost__ __roodevice__  PxPyPzM4D(const PxPyPzM4D & v) :
      fX(v.fX), fY(v.fY), fZ(v.fZ), fM(v.fM) { }

   /**
      assignment operator
    */
   __roohost__ __roodevice__  PxPyPzM4D & operator = (const PxPyPzM4D & v) {
      fX = v.fX;
      fY = v.fY;
      fZ = v.fZ;
      fM = v.fM;
      return *this;
   }


   /**
      construct from any 4D  coordinate system class
      implementing X(), Y(), X() and M()
   */
   template <class AnyCoordSystem>
   __roohost__ __roodevice__  PxPyPzM4D & operator = (const AnyCoordSystem & v) {
      fX = v.X();
      fY = v.Y();
      fZ = v.Z();
      fM = v.M();
      return *this;
   }

   /**
      Set internal data based on an array of 4 Scalar numbers
   */
   __roohost__ __roodevice__  void  SetCoordinates( const Scalar src[] ) {
      fX=src[0]; fY=src[1]; fZ=src[2]; fM=src[3];
      if (fM < 0) RestrictNegMass();
   }

   /**
      get internal data into an array of 4 Scalar numbers
   */
   __roohost__ __roodevice__  void  GetCoordinates( Scalar dest[] ) const
   { dest[0] = fX; dest[1] = fY; dest[2] = fZ; dest[3] = fM; }

   /**
      Set internal data based on 4 Scalar numbers
   */
   __roohost__ __roodevice__  void  SetCoordinates(Scalar  px, Scalar  py, Scalar  pz, Scalar m) {
      fX=px; fY=py; fZ=pz; fM=m;
      if (fM < 0) RestrictNegMass();
   }

   /**
      get internal data into 4 Scalar numbers
   */
   __roohost__ __roodevice__  void  GetCoordinates(Scalar& px, Scalar& py, Scalar& pz, Scalar& m) const
   { px=fX; py=fY; pz=fZ; m=fM;}

   // --------- Coordinates and Coordinate-like Scalar properties -------------

   // cartesian (Minkowski)coordinate accessors

   __roohost__ __roodevice__  Scalar  Px() const { return fX;}
   __roohost__ __roodevice__  Scalar  Py() const { return fY;}
   __roohost__ __roodevice__  Scalar  Pz() const { return fZ;}
   __roohost__ __roodevice__  Scalar  M() const  { return fM; }

   __roohost__ __roodevice__  Scalar  X() const { return fX;}
   __roohost__ __roodevice__  Scalar  Y() const { return fY;}
   __roohost__ __roodevice__  Scalar  Z() const { return fZ;}

   // other coordinate representation
   /**
      Energy
    */
   __roohost__ __roodevice__  Scalar  E() const { return mysqrt(E2()); }

   __roohost__ __roodevice__  Scalar  T() const { return E();}

   /**
      squared magnitude of spatial components
   */
   __roohost__ __roodevice__  Scalar  P2() const { return fX*fX + fY*fY + fZ*fZ; }

   /**
      magnitude of spatial components (magnitude of 3-momentum)
   */
   __roohost__ __roodevice__  Scalar  P() const { return mysqrt(P2()); }
   __roohost__ __roodevice__  Scalar  R() const { return P(); }

   /**
      vector magnitude squared (or mass squared)
      In case of negative mass (spacelike particles return negative values)
   */
   __roohost__ __roodevice__  Scalar  M2() const   {
      return ( fM  >= 0 ) ?  fM*fM :  -fM*fM;
   }
   __roohost__ __roodevice__  Scalar  Mag2() const { return M2(); }

   __roohost__ __roodevice__  Scalar  Mag() const    { return M(); }

   /**
      energy squared
   */
   __roohost__ __roodevice__  Scalar  E2() const {
      Scalar e2 =  P2() + M2();
      // protect against numerical errors when M2() is negative
      return e2 > 0 ? e2 : 0;
   }

   /**
       transverse spatial component squared
   */
   __roohost__ __roodevice__  Scalar  Pt2()   const { return fX*fX + fY*fY;}
   __roohost__ __roodevice__  Scalar  Perp2() const { return Pt2();}

   /**
      Transverse spatial component (P_perp or rho)
   */
   __roohost__ __roodevice__  Scalar  Pt() const { return mysqrt(Perp2()); }
   __roohost__ __roodevice__  Scalar  Perp() const { return Pt();}
   __roohost__ __roodevice__  Scalar  Rho()  const { return Pt();}

   /**
       transverse mass squared
   */
   __roohost__ __roodevice__  Scalar  Mt2() const { return E2() - fZ*fZ; }

   /**
      transverse mass
   */
   __roohost__ __roodevice__  Scalar  Mt() const {
      const Scalar mm = Mt2();
      if (mm >= 0) {
         return mysqrt(mm);
      } else {
         //GenVector::Throw ("PxPyPzM4D::Mt() - Tachyonic:\n"
         //                  "    Pz^2 > E^2 so the transverse mass would be imaginary");
         return -mysqrt(-mm);
      }
   }

   /**
       transverse energy squared
   */
   __roohost__ __roodevice__  Scalar  Et2() const {  // is (E^2 * pt ^2) / p^2
      // but it is faster to form p^2 from pt^2
      Scalar pt2 = Pt2();
      return pt2 == 0 ? 0 : E2() * pt2/( pt2 + fZ*fZ );
   }

   /**
      transverse energy
   */
   __roohost__ __roodevice__  Scalar  Et() const {
      const Scalar etet = Et2();
      return mysqrt(etet);
   }

   /**
      azimuthal angle
   */
   __roohost__ __roodevice__  Scalar  Phi() const { return (fX == 0.0 && fY == 0.0) ? 0.0 : myatan2(fY, fX); }

   /**
      polar angle
   */
   __roohost__ __roodevice__  Scalar  Theta() const { return (fX == 0.0 && fY == 0.0 && fZ == 0.0) ? 0 : myatan2(Pt(), fZ); }

   /**
       pseudorapidity
   */
   __roohost__ __roodevice__  Scalar  Eta() const {
      return Eta_FromRhoZ ( Pt(), fZ);
   }

   // --------- Set Coordinates of this system  ---------------


   /**
      set X value
   */
   __roohost__ __roodevice__  void  SetPx( Scalar  px) {
      fX = px;
   }
   /**
      set Y value
   */
   __roohost__ __roodevice__  void  SetPy( Scalar  py) {
      fY = py;
   }
   /**
      set Z value
   */
   __roohost__ __roodevice__  void  SetPz( Scalar  pz) {
      fZ = pz;
   }
   /**
      set T value
   */
   __roohost__ __roodevice__  void  SetM( Scalar  m) {
      fM = m;
      if (fM < 0) RestrictNegMass();
   }

   /**
       set all values
   */
   __roohost__ __roodevice__  void  SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e);

   // ------ Manipulations -------------

   /**
      negate the 4-vector -  Note that the energy cannot be negate (would need an additional data member)
      therefore negate will work only on the spatial components.
      One would need to use negate only with vectors having the energy as data members
   */
   __roohost__ __roodevice__  void  Negate( ) {
      fX = -fX;
      fY = -fY;
      fZ = -fZ;
      //GenVector::Throw ("PxPyPzM4D::Negate - cannot negate the energy - can negate only the spatial components");
   }

   /**
      scale coordinate values by a scalar quantity a
   */
   __roohost__ __roodevice__  void  Scale( const Scalar & a) {
      fX *= a;
      fY *= a;
      fZ *= a;
      fM *= a;
   }


   /**
      Exact equality
   */
   __roohost__ __roodevice__  bool operator == (const PxPyPzM4D & rhs) const {
      return fX == rhs.fX && fY == rhs.fY && fZ == rhs.fZ && fM == rhs.fM;
   }
   __roohost__ __roodevice__  bool operator != (const PxPyPzM4D & rhs) const {return !(operator==(rhs));}


   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   __roohost__ __roodevice__  Scalar  x() const { return X(); }
   __roohost__ __roodevice__  Scalar  y() const { return Y(); }
   __roohost__ __roodevice__  Scalar  z() const { return Z(); }
   __roohost__ __roodevice__  Scalar  t() const { return E(); }



#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   __roohost__ __roodevice__  void  SetPt(Scalar pt);

   __roohost__ __roodevice__  void  SetEta(Scalar eta);

   __roohost__ __roodevice__  void  SetPhi(Scalar phi);

   __roohost__ __roodevice__  void  SetE(Scalar t);

#endif

private:

   // restrict the value of negative mass to avoid unphysical negative E2 values
   // M2 must be less than P2 for the tachionic particles - otherwise use positive values
    __roohost__ __roodevice__  void  RestrictNegMass() {
      if ( fM >=0 ) return;
      if ( P2() - fM*fM  < 0 ) {
         //GenVector::Throw("PxPyPzM4D::unphysical value of mass, set to closest physical value");
         fM = - P();
      }
      return;
   }


   /**
      (contigous) data containing the coordinate values x,y,z,t
   */

   ScalarType fX;
   ScalarType fY;
   ScalarType fZ;
   ScalarType fM;

};

} // end namespace Experimental
} // end namespace ROOT


// move implementations here to avoid circle dependencies

#include "SYCLMath/GenVector/PxPyPzE4D.h"
#include "SYCLMath/GenVector/PtEtaPhiM4D.h"

#if defined(__MAKECINT__) || defined(G__DICTIONARY)
#include "SYCLMath/GenVector/GenVector_exception.h"
#endif

namespace ROOT {

namespace Experimental {

template <class ScalarType>
 __roohost__ __roodevice__  void  PxPyPzM4D<ScalarType>::SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e) {
   *this = PxPyPzE4D<Scalar> (px, py, pz, e);
}


#if defined(__MAKECINT__) || defined(G__DICTIONARY)

  // ====== Set member functions for coordinates in other systems =======

  // ====== Set member functions for coordinates in other systems =======

template <class ScalarType>
 __roohost__ __roodevice__  void  PxPyPzM4D<ScalarType>::SetPt(ScalarType pt) {
   GenVector_exception e("PxPyPzM4D::SetPt() is not supposed to be called");
   throw e;
   PtEtaPhiE4D<ScalarType> v(*this); v.SetPt(pt); *this = PxPyPzM4D<ScalarType>(v);
}
template <class ScalarType>
 __roohost__ __roodevice__  void  PxPyPzM4D<ScalarType>::SetEta(ScalarType eta) {
   GenVector_exception e("PxPyPzM4D::SetEta() is not supposed to be called");
   throw e;
   PtEtaPhiE4D<ScalarType> v(*this); v.SetEta(eta); *this = PxPyPzM4D<ScalarType>(v);
}
template <class ScalarType>
 __roohost__ __roodevice__  void  PxPyPzM4D<ScalarType>::SetPhi(ScalarType phi) {
   GenVector_exception e("PxPyPzM4D::SetPhi() is not supposed to be called");
   throw e;
   PtEtaPhiE4D<ScalarType> v(*this); v.SetPhi(phi); *this = PxPyPzM4D<ScalarType>(v);
}
template <class ScalarType>
 __roohost__ __roodevice__  void  PxPyPzM4D<ScalarType>::SetE(ScalarType energy) {
   GenVector_exception e("PxPyPzM4D::SetE() is not supposed to be called");
   throw e;
   PxPyPzE4D<ScalarType> v(*this); v.SetE(energy);
   *this = PxPyPzM4D<ScalarType>(v);
}


#endif  // endif __MAKE__CINT || G__DICTIONARY

} // end namespace Experimental

} // end namespace ROOT



#endif // ROOT_Experimental_GenVector_PxPyPzM4D
