// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class PtEtaPhiM4D
//
// Created by: fischler at Wed Jul 21 2005
//   Similar to PtEtaPhiMSystem by moneta
//
// Last update: $Id$
//
#ifndef ROOT_Experimental_GenVector_PtEtaPhiM4D
#define ROOT_Experimental_GenVector_PtEtaPhiM4D  1

#undef __MAKECINT__

#include "SYCLMath/GenVector/etaMax.h"
#include "SYCLMath/GenVector/MathUtil.h"

//#define TRACE_CE
#ifdef TRACE_CE
#include <iostream>
#endif


namespace ROOT {

namespace Experimental {

//__________________________________________________________________________________________
/**
    Class describing a 4D cylindrical coordinate system
    using Pt , Phi, Eta and M (mass)
    The metric used is (-,-,-,+).
    Spacelike particles (M2 < 0) are described with negative mass values,
    but in this case m2 must alwasy be less than P2 to preserve a positive value of E2
    Phi is restricted to be in the range [-PI,PI)

    @ingroup GenVector

    @sa Overview of the @ref GenVector "physics vector library"
*/

template <class ScalarType>
class PtEtaPhiM4D {

public :

   typedef ScalarType Scalar;

   // --------- Constructors ---------------

   /**
      Default constructor gives zero 4-vector (with zero mass)
   */
   __roohost__ __roodevice__  PtEtaPhiM4D() : fPt(0), fEta(0), fPhi(0), fM(0) { }

   /**
      Constructor  from pt, eta, phi, mass values
   */
   __roohost__ __roodevice__  PtEtaPhiM4D(Scalar pt, Scalar eta, Scalar phi, Scalar mass) :
      fPt(pt), fEta(eta), fPhi(phi), fM(mass) {
      RestrictPhi();
      if (fM < 0) RestrictNegMass();
   }

   /**
      Generic constructor from any 4D coordinate system implementing
      Pt(), Eta(), Phi() and M()
   */
   template <class CoordSystem >
   explicit __roohost__ __roodevice__  PtEtaPhiM4D(const CoordSystem & c) :
      fPt(c.Pt()), fEta(c.Eta()), fPhi(c.Phi()), fM(c.M())  { RestrictPhi(); }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // so we decided to re-implement them ( there is no no need to have them with g++4)

   /**
      copy constructor
    */
   __roohost__ __roodevice__  PtEtaPhiM4D(const PtEtaPhiM4D & v) :
      fPt(v.fPt), fEta(v.fEta), fPhi(v.fPhi), fM(v.fM) { }

   /**
      assignment operator
    */
   __roohost__ __roodevice__  PtEtaPhiM4D & operator = (const PtEtaPhiM4D & v) {
      fPt  = v.fPt;
      fEta = v.fEta;
      fPhi = v.fPhi;
      fM   = v.fM;
      return *this;
   }


   /**
      Set internal data based on an array of 4 Scalar numbers
   */
   __roohost__ __roodevice__ void SetCoordinates( const Scalar src[] ) {
      fPt=src[0]; fEta=src[1]; fPhi=src[2]; fM=src[3];
      RestrictPhi();
      if (fM <0) RestrictNegMass();
   }

   /**
      get internal data into an array of 4 Scalar numbers
   */
   __roohost__ __roodevice__ void GetCoordinates( Scalar dest[] ) const
   { dest[0] = fPt; dest[1] = fEta; dest[2] = fPhi; dest[3] = fM; }

   /**
      Set internal data based on 4 Scalar numbers
   */
   __roohost__ __roodevice__ void SetCoordinates(Scalar pt, Scalar eta, Scalar phi, Scalar mass) {
      fPt=pt; fEta = eta; fPhi = phi; fM = mass;
      RestrictPhi();
      if (fM <0) RestrictNegMass();
   }

   /**
      get internal data into 4 Scalar numbers
   */
   __roohost__ __roodevice__ void
   GetCoordinates(Scalar& pt, Scalar & eta, Scalar & phi, Scalar& mass) const
   { pt=fPt; eta=fEta; phi = fPhi; mass = fM; }

   // --------- Coordinates and Coordinate-like Scalar properties -------------

   // 4-D Cylindrical eta coordinate accessors

   __roohost__ __roodevice__ Scalar Pt()  const { return fPt;  }
   __roohost__ __roodevice__ Scalar Eta() const { return fEta; }
   __roohost__ __roodevice__ Scalar Phi() const { return fPhi; }
   /**
       M() is the invariant mass;
       in this coordinate system it can be negagative if set that way.
   */
   __roohost__ __roodevice__ Scalar  M()   const { return fM;   }
   __roohost__ __roodevice__ Scalar Mag() const { return M(); }

   __roohost__ __roodevice__ Scalar Perp()const { return Pt(); }
   __roohost__ __roodevice__ Scalar Rho() const { return Pt(); }

   // other coordinate representation

   __roohost__ __roodevice__ Scalar Px() const { return fPt * mycos(fPhi); }
   __roohost__ __roodevice__ Scalar X () const { return Px();         }
   __roohost__ __roodevice__ Scalar Py() const { return fPt * mysin(fPhi); }
   __roohost__ __roodevice__ Scalar Y () const { return Py();         }
   __roohost__ __roodevice__ Scalar Pz() const {
      return fPt > 0 ? fPt * mysinh(fEta) : fEta == 0 ? 0 : fEta > 0 ? fEta - etaMax<Scalar>() : fEta + etaMax<Scalar>();
   }
   __roohost__ __roodevice__ Scalar Z () const { return Pz(); }

   /**
       magnitude of momentum
   */
   __roohost__ __roodevice__ Scalar P() const {
      return fPt > 0 ? fPt * mycosh(fEta)
                     : fEta > etaMax<Scalar>() ? fEta - etaMax<Scalar>()
                                               : fEta < -etaMax<Scalar>() ? -fEta - etaMax<Scalar>() : 0;
   }
   __roohost__ __roodevice__ Scalar R() const { return P(); }

   /**
       squared magnitude of spatial components (momentum squared)
   */
   __roohost__ __roodevice__ Scalar P2() const
   {
      const Scalar p = P();
      return p * p;
   }

   /**
       energy squared
   */
   __roohost__ __roodevice__ Scalar E2() const {
      Scalar e2 =  P2() + M2();
      // avoid rounding error which can make E2 negative when M2 is negative
      return e2 > 0 ? e2 : 0;
   }

   /**
       Energy (timelike component of momentum-energy 4-vector)
   */
   __roohost__ __roodevice__ Scalar E() const { return mysqrt(E2()); }

   __roohost__ __roodevice__ Scalar T()   const { return E();  }

   /**
      vector magnitude squared (or mass squared)
      In case of negative mass (spacelike particles return negative values)
   */
   __roohost__ __roodevice__ Scalar M2() const   {
      return ( fM  >= 0 ) ?  fM*fM :  -fM*fM;
   }
   __roohost__ __roodevice__ Scalar Mag2() const { return M2();  }

   /**
       transverse spatial component squared
   */
   __roohost__ __roodevice__ Scalar Pt2()   const { return fPt*fPt;}
   __roohost__ __roodevice__ Scalar Perp2() const { return Pt2();  }

   /**
       transverse mass squared
   */
   __roohost__ __roodevice__ Scalar Mt2() const { return M2()  + fPt*fPt; }

   /**
      transverse mass - will be negative if Mt2() is negative
   */
   __roohost__ __roodevice__ Scalar Mt() const {
      const Scalar mm = Mt2();
      if (mm >= 0) {
         return mysqrt(mm);
      } else {
         //GenVector::Throw  ("PtEtaPhiM4D::Mt() - Tachyonic:\n"
         //                   "    Pz^2 > E^2 so the transverse mass would be imaginary");
         return -mysqrt(-mm);
      }
   }

   /**
       transverse energy squared
   */
   __roohost__ __roodevice__ Scalar Et2() const {
      // a bit faster than et * et
      return 2. * E2() / (mycosh(2 * fEta) + 1);
   }

   /**
      transverse energy
   */
   __roohost__ __roodevice__ Scalar Et() const { return E() / mycosh(fEta); }

private:
   inline static __roohost__ __roodevice__ Scalar pi() { return M_PI; }
   inline __roohost__ __roodevice__ void RestrictPhi() {
      if (fPhi <= -pi() || fPhi > pi()) fPhi = fPhi - myfloor(fPhi / (2 * pi()) + .5) * 2 * pi();
   }
   // restrict the value of negative mass to avoid unphysical negative E2 values
   // M2 must be less than P2 for the tachionic particles - otherwise use positive values
   inline __roohost__ __roodevice__ void RestrictNegMass() {
      if (fM < 0) {
         if (P2() - fM * fM < 0) {
            //GenVector::Throw("PtEtaPhiM4D::unphysical value of mass, set to closest physical value");
            fM = -P();
         }
      }
   }

public:

   /**
      polar angle
   */
   __roohost__ __roodevice__ Scalar Theta() const { return (fPt > 0 ? Scalar(2) * myatan(myexp(-fEta)) : fEta >= 0 ? 0 : pi()); }

   // --------- Set Coordinates of this system  ---------------

   /**
      set Pt value
   */
   __roohost__ __roodevice__ void SetPt( Scalar  pt) {
      fPt = pt;
   }
   /**
      set eta value
   */
   __roohost__ __roodevice__ void SetEta( Scalar  eta) {
      fEta = eta;
   }
   /**
      set phi value
   */
   __roohost__ __roodevice__ void SetPhi( Scalar  phi) {
      fPhi = phi;
      RestrictPhi();
   }
   /**
      set M value
   */
   __roohost__ __roodevice__ void SetM( Scalar  mass) {
      fM = mass;
      if (fM <0) RestrictNegMass();
   }

   /**
       set values using cartesian coordinate system
   */
   __roohost__ __roodevice__ void SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e);


   // ------ Manipulations -------------

   /**
      negate the 4-vector -- Note that the energy cannot be negate (would need an additional data member)
      therefore negate will work only on the spatial components
      One would need to use negate only with vectors having the energy as data members
   */
   __roohost__ __roodevice__ void Negate( ) {
      fPhi = ( (fPhi > 0) ? fPhi - pi() : fPhi + pi()  );
      fEta = - fEta;
      //GenVector::Throw ("PtEtaPhiM4D::Negate - cannot negate the energy - can negate only the spatial components");
   }

   /**
      Scale coordinate values by a scalar quantity a
   */
   __roohost__ __roodevice__ void Scale( Scalar a) {
      if (a < 0) {
         Negate(); a = -a;
      }
      fPt *= a;
      fM  *= a;
   }

   /**
      Assignment from a generic coordinate system implementing
      Pt(), Eta(), Phi() and M()
   */
   template <class CoordSystem >
   PtEtaPhiM4D & operator = (const CoordSystem & c) {
      fPt  = c.Pt();
      fEta = c.Eta();
      fPhi = c.Phi();
      fM   = c.M();
      return *this;
   }

   /**
      Exact equality
   */
   bool operator == (const PtEtaPhiM4D & rhs) const {
      return fPt == rhs.fPt && fEta == rhs.fEta
         && fPhi == rhs.fPhi && fM == rhs.fM;
   }
   bool operator != (const PtEtaPhiM4D & rhs) const {return !(operator==(rhs));}

   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   __roohost__ __roodevice__ Scalar x() const { return X(); }
   __roohost__ __roodevice__ Scalar y() const { return Y(); }
   __roohost__ __roodevice__ Scalar z() const { return Z(); }
   __roohost__ __roodevice__ Scalar t() const { return E(); }


#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   __roohost__ __roodevice__ void SetPx(Scalar px);

   __roohost__ __roodevice__ void SetPy(Scalar py);

   __roohost__ __roodevice__ void SetPz(Scalar pz);

   __roohost__ __roodevice__ void SetE(Scalar t);

#endif

private:

   ScalarType fPt;
   ScalarType fEta;
   ScalarType fPhi;
   ScalarType fM;

};


} // end namespace Experimental
} // end namespace ROOT


// move implementations here to avoid circle dependencies
#include "SYCLMath/GenVector/PxPyPzE4D.h"

#if defined(__MAKECINT__) || defined(G__DICTIONARY)
#include "SYCLMath/GenVector/GenVector_exception.h"
#endif

namespace ROOT {

namespace Experimental {


template <class ScalarType>
inline __roohost__ __roodevice__ void PtEtaPhiM4D<ScalarType>::SetPxPyPzE(Scalar px, Scalar py, Scalar pz, Scalar e) {
   *this = PxPyPzE4D<Scalar> (px, py, pz, e);
}


#if defined(__MAKECINT__) || defined(G__DICTIONARY)

  // ====== Set member functions for coordinates in other systems =======

template <class ScalarType>
__roohost__ __roodevice__ void PtEtaPhiM4D<ScalarType>::SetPx(Scalar px) {
   GenVector_exception e("PtEtaPhiM4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPx(px); *this = PtEtaPhiM4D<Scalar>(v);
}
template <class ScalarType>
__roohost__ __roodevice__ void PtEtaPhiM4D<ScalarType>::SetPy(Scalar py) {
   GenVector_exception e("PtEtaPhiM4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPy(py); *this = PtEtaPhiM4D<Scalar>(v);
}
template <class ScalarType>
__roohost__ __roodevice__ void PtEtaPhiM4D<ScalarType>::SetPz(Scalar pz) {
   GenVector_exception e("PtEtaPhiM4D::SetPx() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetPz(pz); *this = PtEtaPhiM4D<Scalar>(v);
}
template <class ScalarType>
__roohost__ __roodevice__ void PtEtaPhiM4D<ScalarType>::SetE(Scalar energy) {
   GenVector_exception e("PtEtaPhiM4D::SetE() is not supposed to be called");
   throw e;
   PxPyPzE4D<Scalar> v(*this); v.SetE(energy);   *this = PtEtaPhiM4D<Scalar>(v);
}

#endif  // endif __MAKE__CINT || G__DICTIONARY

} // end namespace Experimental

} // end namespace ROOT



#endif // ROOT_Experimental_GenVector_PtEtaPhiM4D
