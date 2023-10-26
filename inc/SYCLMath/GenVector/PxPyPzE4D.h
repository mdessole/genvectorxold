// @(#)root/mathcore:$Id: 04c6d98020d7178ed5f0884f9466bca32b031565 $
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT MathLib Team                         *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class PxPyPzE4D
//
// Created by: fischler at Wed Jul 20   2005
//   (starting from PxPyPzE4D by moneta)
//
// Last update: $Id: 04c6d98020d7178ed5f0884f9466bca32b031565 $
//
#ifndef ROOT_Experimental_GenVector_PxPyPzE4D
#define ROOT_Experimental_GenVector_PxPyPzE4D  1

#undef __MAKECINT__

#include "SYCLMath/GenVector/MathUtil.h"


namespace ROOT {

namespace Experimental {

//__________________________________________________________________________________________
/**
    Class describing a 4D cartesian coordinate system (x, y, z, t coordinates)
    or momentum-energy vectors stored as (Px, Py, Pz, E).
    The metric used is (-,-,-,+)

    @ingroup GenVector

    @sa Overview of the @ref GenVector "physics vector library"
*/

template <class ScalarType = double>
class PxPyPzE4D {

public :

   typedef ScalarType Scalar;

   // --------- Constructors ---------------

   /**
      Default constructor  with x=y=z=t=0
   */
   __roohost__ __roodevice__ PxPyPzE4D() : fX(0.0), fY(0.0), fZ(0.0), fT(0.0) { }


   /**
      Constructor  from x, y , z , t values
   */
   __roohost__ __roodevice__ PxPyPzE4D(Scalar px, Scalar py, Scalar pz, Scalar e) :
      fX(px), fY(py), fZ(pz), fT(e) { }


   /**
      construct from any vector or  coordinate system class
      implementing x(), y() and z() and t()
   */
   template <class CoordSystem>
   explicit __roohost__ __roodevice__  PxPyPzE4D(const CoordSystem & v) :
      fX( v.x() ), fY( v.y() ), fZ( v.z() ), fT( v.t() )  { }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // so we decided to re-implement them ( there is no no need to have them with g++4)
   /**
      copy constructor
    */
   __roohost__ __roodevice__ PxPyPzE4D(const PxPyPzE4D & v) :
      fX(v.fX), fY(v.fY), fZ(v.fZ), fT(v.fT) { }

   /**
      assignment operator
    */
   __roohost__ __roodevice__ PxPyPzE4D & operator = (const PxPyPzE4D & v) {
      fX = v.fX;
      fY = v.fY;
      fZ = v.fZ;
      fT = v.fT;
      return *this;
   }

   /**
      Set internal data based on an array of 4 Scalar numbers
   */
   __roohost__ __roodevice__ void SetCoordinates( const Scalar src[] )
   { fX=src[0]; fY=src[1]; fZ=src[2]; fT=src[3]; }

   /**
      get internal data into an array of 4 Scalar numbers
   */
   __roohost__ __roodevice__ void GetCoordinates( Scalar dest[] ) const
   { dest[0] = fX; dest[1] = fY; dest[2] = fZ; dest[3] = fT; }

   /**
      Set internal data based on 4 Scalar numbers
   */
   __roohost__ __roodevice__ void SetCoordinates(Scalar  px, Scalar  py, Scalar  pz, Scalar e)
   { fX=px; fY=py; fZ=pz; fT=e;}

   /**
      get internal data into 4 Scalar numbers
   */
   __roohost__ __roodevice__ void GetCoordinates(Scalar& px, Scalar& py, Scalar& pz, Scalar& e) const
   { px=fX; py=fY; pz=fZ; e=fT;}

   // --------- Coordinates and Coordinate-like Scalar properties -------------

   // cartesian (Minkowski)coordinate accessors

   __roohost__ __roodevice__ Scalar Px() const { return fX;}
   __roohost__ __roodevice__ Scalar Py() const { return fY;}
   __roohost__ __roodevice__ Scalar Pz() const { return fZ;}
   __roohost__ __roodevice__ Scalar E()  const { return fT;}

   __roohost__ __roodevice__ Scalar X() const { return fX;}
   __roohost__ __roodevice__ Scalar Y() const { return fY;}
   __roohost__ __roodevice__ Scalar Z() const { return fZ;}
   __roohost__ __roodevice__ Scalar T() const { return fT;}

   // other coordinate representation

   /**
      squared magnitude of spatial components
   */
   __roohost__ __roodevice__ Scalar P2() const { return fX*fX + fY*fY + fZ*fZ; }

   /**
      magnitude of spatial components (magnitude of 3-momentum)
   */
   __roohost__ __roodevice__ Scalar P() const {  return mysqrt(P2()); }
   __roohost__ __roodevice__ Scalar R() const { return P(); }

   /**
      vector magnitude squared (or mass squared)
   */
   __roohost__ __roodevice__ Scalar M2() const   { return fT*fT - fX*fX - fY*fY - fZ*fZ;}
   __roohost__ __roodevice__ Scalar Mag2() const { return M2(); }

   /**
      invariant mass
   */
   __roohost__ __roodevice__ Scalar M() const
   {
      const Scalar mm = M2();
      if (mm >= 0) {
         return mysqrt(mm);
      } else {
         //GenVector::Throw ("PxPyPzE4D::M() - Tachyonic:\n"
         //          "    P^2 > E^2 so the mass would be imaginary");
         return -mysqrt(-mm);
      }
   }
   __roohost__ __roodevice__ Scalar Mag() const    { return M(); }

   /**
       transverse spatial component squared
   */
   __roohost__ __roodevice__ Scalar Pt2()   const { return fX*fX + fY*fY;}
   __roohost__ __roodevice__ Scalar Perp2() const { return Pt2();}

   /**
      Transverse spatial component (P_perp or rho)
   */
   __roohost__ __roodevice__ Scalar Pt() const { return mysqrt(Perp2()); }
   __roohost__ __roodevice__ Scalar Perp() const { return Pt();}
   __roohost__ __roodevice__ Scalar Rho()  const { return Pt();}

   /**
       transverse mass squared
   */
   __roohost__ __roodevice__ Scalar Mt2() const { return fT*fT - fZ*fZ; }

   /**
      transverse mass
   */
   __roohost__ __roodevice__ Scalar Mt() const {
      const Scalar mm = Mt2();
      if (mm >= 0) {
         return mysqrt(mm);
      } else {
         //GenVector::Throw ("PxPyPzE4D::Mt() - Tachyonic:\n"
         //                  "    Pz^2 > E^2 so the transverse mass would be imaginary");
         return -mysqrt(-mm);
      }
   }

   /**
       transverse energy squared
   */
   __roohost__ __roodevice__ Scalar Et2() const {  // is (E^2 * pt ^2) / p^2
      // but it is faster to form p^2 from pt^2
      Scalar pt2 = Pt2();
      return pt2 == 0 ? 0 : fT*fT * pt2/( pt2 + fZ*fZ );
   }

   /**
      transverse energy
   */
   __roohost__ __roodevice__ Scalar Et() const {
      const Scalar etet = Et2();
      return fT < 0.0 ? -mysqrt(etet) : mysqrt(etet);
   }

   /**
      azimuthal angle
   */
   __roohost__ __roodevice__ Scalar Phi() const { return (fX == 0.0 && fY == 0.0) ? 0 : myatan2(fY, fX); }

   /**
      polar angle
   */
   __roohost__ __roodevice__ Scalar Theta() const { return (fX == 0.0 && fY == 0.0 && fZ == 0.0) ? 0 : myatan2(Pt(), fZ); }

   /**
       pseudorapidity
   */
   __roohost__ __roodevice__ Scalar Eta() const {
      return Eta_FromRhoZ ( Pt(), fZ);
   }

   // --------- Set Coordinates of this system  ---------------


   /**
      set X value
   */
   __roohost__ __roodevice__ void SetPx( Scalar  px) {
      fX = px;
   }
   /**
      set Y value
   */
   __roohost__ __roodevice__ void SetPy( Scalar  py) {
      fY = py;
   }
   /**
      set Z value
   */
   __roohost__ __roodevice__ void SetPz( Scalar  pz) {
      fZ = pz;
   }
   /**
      set T value
   */
   __roohost__ __roodevice__ void SetE( Scalar  e) {
      fT = e;
   }

   /**
       set all values using cartesian coordinates
   */
   __roohost__ __roodevice__ void SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e) {
      fX=px;
      fY=py;
      fZ=pz;
      fT=e;
   }



   // ------ Manipulations -------------

   /**
      negate the 4-vector
   */
   __roohost__ __roodevice__ void Negate( ) { fX = -fX; fY = -fY;  fZ = -fZ; fT = -fT;}

   /**
      scale coordinate values by a scalar quantity a
   */
   __roohost__ __roodevice__ void Scale( const Scalar & a) {
      fX *= a;
      fY *= a;
      fZ *= a;
      fT *= a;
   }

   /**
      Assignment from a generic coordinate system implementing
      x(), y(), z() and t()
   */
   template <class AnyCoordSystem>
   PxPyPzE4D & operator = (const AnyCoordSystem & v) {
      fX = v.x();
      fY = v.y();
      fZ = v.z();
      fT = v.t();
      return *this;
   }

   /**
      Exact equality
   */
   bool operator == (const PxPyPzE4D & rhs) const {
      return fX == rhs.fX && fY == rhs.fY && fZ == rhs.fZ && fT == rhs.fT;
   }
   bool operator != (const PxPyPzE4D & rhs) const {return !(operator==(rhs));}


   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   __roohost__ __roodevice__ Scalar x() const { return fX; }
   __roohost__ __roodevice__ Scalar y() const { return fY; }
   __roohost__ __roodevice__ Scalar z() const { return fZ; }
   __roohost__ __roodevice__ Scalar t() const { return fT; }



#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   __roohost__ __roodevice__ void SetPt(Scalar pt);

   __roohost__ __roodevice__ void SetEta(Scalar eta);

   __roohost__ __roodevice__ void SetPhi(Scalar phi);

   __roohost__ __roodevice__ void SetM(Scalar m);

#endif

private:

   /**
      (contigous) data containing the coordinate values x,y,z,t
   */

   ScalarType fX;
   ScalarType fY;
   ScalarType fZ;
   ScalarType fT;

};

} // end namespace Experimental
} // end namespace ROOT



#if defined(__MAKECINT__) || defined(G__DICTIONARY)
// move implementations here to avoid circle dependencies

#include "SYCLMath/GenVector/PtEtaPhiE4D.h"
#include "SYCLMath/GenVector/PtEtaPhiM4D.h"

#include "SYCLMath/GenVector/GenVector_exception.h"

namespace ROOT {

namespace Experimental {


    // ====== Set member functions for coordinates in other systems =======
    // throw always exceptions  in this case

template <class ScalarType>
__roohost__ __roodevice__ void PxPyPzE4D<ScalarType>::SetPt(Scalar pt) {
   GenVector_exception e("PxPyPzE4D::SetPt() is not supposed to be called");
   throw e;
   PtEtaPhiE4D<Scalar> v(*this); v.SetPt(pt); *this = PxPyPzE4D<Scalar>(v);
}
template <class ScalarType>
__roohost__ __roodevice__ void PxPyPzE4D<ScalarType>::SetEta(Scalar eta) {
   GenVector_exception e("PxPyPzE4D::SetEta() is not supposed to be called");
   throw e;
   PtEtaPhiE4D<Scalar> v(*this); v.SetEta(eta); *this = PxPyPzE4D<Scalar>(v);
}
template <class ScalarType>
__roohost__ __roodevice__ void PxPyPzE4D<ScalarType>::SetPhi(Scalar phi) {
   GenVector_exception e("PxPyPzE4D::SetPhi() is not supposed to be called");
   throw e;
   PtEtaPhiE4D<Scalar> v(*this); v.SetPhi(phi); *this = PxPyPzE4D<Scalar>(v);
}

template <class ScalarType>
__roohost__ __roodevice__ void PxPyPzE4D<ScalarType>::SetM(Scalar m) {
   GenVector_exception e("PxPyPzE4D::SetM() is not supposed to be called");
   throw e;
   PtEtaPhiM4D<Scalar> v(*this); v.SetM(m);
   *this = PxPyPzE4D<Scalar>(v);
}


} // end namespace Experimental

} // end namespace ROOT

#endif  // endif __MAKE__CINT || G__DICTIONARY


#endif // ROOT_Experimental_GenVector_PxPyPzE4D
