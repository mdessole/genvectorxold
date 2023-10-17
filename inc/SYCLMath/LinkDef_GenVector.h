// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005



#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace ROOT;
#pragma link C++ namespace ROOT::Experimental;


#pragma link C++ class    ROOT::Experimental::Cartesian2D<double>+;
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<Double32_t>" \
             targetClass="ROOT::Experimental::Cartesian2D<double>";
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<float>"      \
             targetClass="ROOT::Experimental::Cartesian2D<double>";
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<Float16_t>"  \
             targetClass="ROOT::Experimental::Cartesian2D<double>";

#pragma link C++ class    ROOT::Experimental::Polar2D<double>+;
#pragma read sourceClass="ROOT::Experimental::Polar2D<Double32_t>" \
             targetClass="ROOT::Experimental::Polar2D<double>";
#pragma read sourceClass="ROOT::Experimental::Polar2D<float>"      \
             targetClass="ROOT::Experimental::Polar2D<double>";
#pragma read sourceClass="ROOT::Experimental::Polar2D<Float16_t>"  \
             targetClass="ROOT::Experimental::Polar2D<double>";



#pragma link C++ class    ROOT::Experimental::Cartesian3D<double>+;
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<Double32_t>" \
             targetClass="ROOT::Experimental::Cartesian3D<double>";
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<float>"      \
             targetClass="ROOT::Experimental::Cartesian3D<double>";
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<Float16_t>"  \
             targetClass="ROOT::Experimental::Cartesian3D<double>";

#pragma link C++ class    ROOT::Experimental::Polar3D<double>+;
#pragma read sourceClass="ROOT::Experimental::Polar3D<Double32_t>" \
             targetClass="ROOT::Experimental::Polar3D<double>";
#pragma read sourceClass="ROOT::Experimental::Polar3D<float>"      \
             targetClass="ROOT::Experimental::Polar3D<double>";
#pragma read sourceClass="ROOT::Experimental::Polar3D<Float16_t>"  \
             targetClass="ROOT::Experimental::Polar3D<double>";

#pragma link C++ class    ROOT::Experimental::Cylindrical3D<double>+;
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<Double32_t>" \
             targetClass="ROOT::Experimental::Cylindrical3D<double>";
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<float>"      \
             targetClass="ROOT::Experimental::Cylindrical3D<double>";
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<Float16_t>"  \
             targetClass="ROOT::Experimental::Cylindrical3D<double>";

#pragma link C++ class    ROOT::Experimental::CylindricalEta3D<double>+;
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<Double32_t>" \
             targetClass="ROOT::Experimental::CylindricalEta3D<double>";
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<float>"      \
             targetClass="ROOT::Experimental::CylindricalEta3D<double>";
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<Float16_t>"  \
             targetClass="ROOT::Experimental::CylindricalEta3D<double>";


#pragma link C++ class ROOT::Experimental::DefaultCoordinateSystemTag+;
#pragma link C++ class ROOT::Experimental::LocalCoordinateSystemTag+;
#pragma link C++ class ROOT::Experimental::GlobalCoordinateSystemTag+;

#pragma link C++ class    ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<double> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<float> >"      \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<double> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<double> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<float> >"      \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<double> >";


#pragma link C++ class    ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<double> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<float> >"      \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<double> >";

#pragma link C++ class    ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<double> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<float> >"      \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<double> >";



#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float> >"      \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float> >"      \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float> >"      \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float> >"      \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double> >";


#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float> >"      \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float> >"      \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float> >"      \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float> >"      \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double> >";


#ifdef __CLING__
// Work around CINT and autoloader deficiency with template default parameter
// Those requests are solely for rlibmap, they do no need to be seen by rootcint.
#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";


#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >"      \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#endif

#pragma link C++ class    ROOT::Experimental::PxPyPzE4D<double>+;
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<Double32_t>" \
             targetClass="ROOT::Experimental::PxPyPzE4D<double>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<float>"      \
             targetClass="ROOT::Experimental::PxPyPzE4D<double>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<Float16_t>"  \
             targetClass="ROOT::Experimental::PxPyPzE4D<double>";

#pragma link C++ class    ROOT::Experimental::PtEtaPhiE4D<double>+;
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<Double32_t>" \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<double>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<float>"      \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<double>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<Float16_t>"  \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<double>";

#pragma link C++ class    ROOT::Experimental::PxPyPzM4D<double>+;
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<Double32_t>" \
             targetClass="ROOT::Experimental::PxPyPzM4D<double>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<float>"      \
             targetClass="ROOT::Experimental::PxPyPzM4D<double>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<Float16_t>"  \
             targetClass="ROOT::Experimental::PxPyPzM4D<double>";

#pragma link C++ class    ROOT::Experimental::PtEtaPhiM4D<double>+;
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<Double32_t>" \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<double>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<float>"      \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<double>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<Float16_t>"  \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<double>";

//#pragma link C++ class    ROOT::Experimental::EEtaPhiMSystem<double>+;
#pragma read sourceClass="ROOT::Experimental::EEtaPhiMSystem<Double32_t>" \
             targetClass="ROOT::Experimental::EEtaPhiMSystem<double>";
#pragma read sourceClass="ROOT::Experimental::EEtaPhiMSystem<float>"      \
             targetClass="ROOT::Experimental::EEtaPhiMSystem<double>";
#pragma read sourceClass="ROOT::Experimental::EEtaPhiMSystem<Float16_t>"  \
             targetClass="ROOT::Experimental::EEtaPhiMSystem<double>";

//#pragma link C++ class    ROOT::Experimental::PtEtaPhiMSystem<double>+;
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiMSystem<Double32_t>" \
             targetClass="ROOT::Experimental::PtEtaPhiMSystem<double>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiMSystem<float>"      \
             targetClass="ROOT::Experimental::PtEtaPhiMSystem<double>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiMSystem<Float16_t>"  \
             targetClass="ROOT::Experimental::PtEtaPhiMSystem<double>";


#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<double> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Double32_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<double> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<float> >"      \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<double> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Float16_t> >"  \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<double> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<double> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Double32_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<double> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<float> >"      \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<double> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Float16_t> >"  \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<double> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<double> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Double32_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<double> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<float> >"      \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<double> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Float16_t> >"  \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<double> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<double> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Double32_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<double> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<float> >"      \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<double> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Float16_t> >"  \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<double> >";


//// Floating types 

#pragma link C++ class    ROOT::Experimental::Cartesian2D<float>+;
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<double>"     \
             targetClass="ROOT::Experimental::Cartesian2D<float>";
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<Double32_t>" \
             targetClass="ROOT::Experimental::Cartesian2D<float>";
#pragma read sourceClass="ROOT::Experimental::Cartesian2D<Float16_t>"  \
             targetClass="ROOT::Experimental::Cartesian2D<float>";

#pragma link C++ class    ROOT::Experimental::Polar2D<float>+;
#pragma read sourceClass="ROOT::Experimental::Polar2D<double>"     \
             targetClass="ROOT::Experimental::Polar2D<float>";
#pragma read sourceClass="ROOT::Experimental::Polar2D<Double32_t>" \
             targetClass="ROOT::Experimental::Polar2D<float>";
#pragma read sourceClass="ROOT::Experimental::Polar2D<Float16_t>"  \
             targetClass="ROOT::Experimental::Polar2D<float>";


#pragma link C++ class    ROOT::Experimental::Cartesian3D<float>+;
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<double>"     \
             targetClass="ROOT::Experimental::Cartesian3D<float>";
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<Double32_t>" \
             targetClass="ROOT::Experimental::Cartesian3D<float>";
#pragma read sourceClass="ROOT::Experimental::Cartesian3D<Float16_t>"  \
             targetClass="ROOT::Experimental::Cartesian3D<float>";

#pragma link C++ class    ROOT::Experimental::Polar3D<float>+;
#pragma read sourceClass="ROOT::Experimental::Polar3D<double>"     \
             targetClass="ROOT::Experimental::Polar3D<float>";
#pragma read sourceClass="ROOT::Experimental::Polar3D<Double32_t>" \
             targetClass="ROOT::Experimental::Polar3D<float>";
#pragma read sourceClass="ROOT::Experimental::Polar3D<Float16_t>"  \
             targetClass="ROOT::Experimental::Polar3D<float>";

#pragma link C++ class    ROOT::Experimental::Cylindrical3D<float>+;
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<double>"     \
             targetClass="ROOT::Experimental::Cylindrical3D<float>";
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<Double32_t>" \
             targetClass="ROOT::Experimental::Cylindrical3D<float>";
#pragma read sourceClass="ROOT::Experimental::Cylindrical3D<Float16_t>"  \
             targetClass="ROOT::Experimental::Cylindrical3D<float>";

#pragma link C++ class    ROOT::Experimental::CylindricalEta3D<float>+;
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<double>"     \
             targetClass="ROOT::Experimental::CylindricalEta3D<float>";
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<Double32_t>" \
             targetClass="ROOT::Experimental::CylindricalEta3D<float>";
#pragma read sourceClass="ROOT::Experimental::CylindricalEta3D<Float16_t>"  \
             targetClass="ROOT::Experimental::CylindricalEta3D<float>";


#pragma link C++ class ROOT::Experimental::DefaultCoordinateSystemTag+;
#pragma link C++ class ROOT::Experimental::LocalCoordinateSystemTag+;
#pragma link C++ class ROOT::Experimental::GlobalCoordinateSystemTag+;

#pragma link C++ class    ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<float> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<double> >"     \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Cartesian2D<float> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<float> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<double> >"     \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector2D<ROOT::Experimental::Polar2D<float> >";


#pragma link C++ class    ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<float> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<double> >"     \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Cartesian2D<float> >";

#pragma link C++ class    ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<float> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<double> >"     \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector2D<ROOT::Experimental::Polar2D<float> >";



#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double> >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double> >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double> >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float> >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float> >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double> >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float> >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float> >";


#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double> >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double> >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double> >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float> >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float> >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double> >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t> >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float> >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t> >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float> >";


#ifdef __CLING__
// Work around CINT and autoloader deficiency with template default parameter
// Those requests are solely for rlibmap, they do no need to be seen by rootcint.
#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::DisplacementVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";


#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cartesian3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Polar3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::Cylindrical3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#pragma link C++ class    ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >+;
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<double>,ROOT::Experimental::DefaultCoordinateSystemTag >"     \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Double32_t>,ROOT::Experimental::DefaultCoordinateSystemTag >" \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";
#pragma read sourceClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<Float16_t>,ROOT::Experimental::DefaultCoordinateSystemTag >"  \
             targetClass="ROOT::Experimental::PositionVector3D<ROOT::Experimental::CylindricalEta3D<float>,ROOT::Experimental::DefaultCoordinateSystemTag >";

#endif

#pragma link C++ class    ROOT::Experimental::PxPyPzE4D<float>+;
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<double>"     \
             targetClass="ROOT::Experimental::PxPyPzE4D<float>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<Double32_t>" \
             targetClass="ROOT::Experimental::PxPyPzE4D<float>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzE4D<Float16_t>"  \
             targetClass="ROOT::Experimental::PxPyPzE4D<float>";

#pragma link C++ class    ROOT::Experimental::PtEtaPhiE4D<float>+;
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<double>"     \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<float>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<Double32_t>" \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<float>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiE4D<Float16_t>"  \
             targetClass="ROOT::Experimental::PtEtaPhiE4D<float>";

#pragma link C++ class    ROOT::Experimental::PxPyPzM4D<float>+;
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<double>"     \
             targetClass="ROOT::Experimental::PxPyPzM4D<float>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<Double32_t>" \
             targetClass="ROOT::Experimental::PxPyPzM4D<float>";
#pragma read sourceClass="ROOT::Experimental::PxPyPzM4D<Float16_t>"  \
             targetClass="ROOT::Experimental::PxPyPzM4D<float>";

#pragma link C++ class    ROOT::Experimental::PtEtaPhiM4D<float>+;
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<double>"     \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<float>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<Double32_t>" \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<float>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiM4D<Float16_t>"  \
             targetClass="ROOT::Experimental::PtEtaPhiM4D<float>";

//#pragma link C++ class    ROOT::Experimental::EEtaPhiMSystem<float>+;
#pragma read sourceClass="ROOT::Experimental::EEtaPhiMSystem<double>"     \
             targetClass="ROOT::Experimental::EEtaPhiMSystem<float>";
#pragma read sourceClass="ROOT::Experimental::EEtaPhiMSystem<Double32_t>" \
             targetClass="ROOT::Experimental::EEtaPhiMSystem<float>";
#pragma read sourceClass="ROOT::Experimental::EEtaPhiMSystem<Float16_t>"  \
             targetClass="ROOT::Experimental::EEtaPhiMSystem<float>";

//#pragma link C++ class    ROOT::Experimental::PtEtaPhiMSystem<float>+;
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiMSystem<double>"     \
             targetClass="ROOT::Experimental::PtEtaPhiMSystem<float>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiMSystem<Double32_t>" \
             targetClass="ROOT::Experimental::PtEtaPhiMSystem<float>";
#pragma read sourceClass="ROOT::Experimental::PtEtaPhiMSystem<Float16_t>"  \
             targetClass="ROOT::Experimental::PtEtaPhiMSystem<float>";


#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<float> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<double> >"     \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<float> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Double32_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<float> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<Float16_t> >"  \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzE4D<float> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<float> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<double> >"     \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<float> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Double32_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<float> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<Float16_t> >"  \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiE4D<float> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<float> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<double> >"     \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<float> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Double32_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<float> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<Float16_t> >"  \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PxPyPzM4D<float> >";

#pragma link C++ class    ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<float> >+;
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<double> >"     \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<float> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Double32_t> >" \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<float> >";
#pragma read sourceClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<Float16_t> >"  \
             targetClass="ROOT::Experimental::LorentzVector<ROOT::Experimental::PtEtaPhiM4D<float> >";




// rotations
//#ifdef LATER

#pragma link C++ class ROOT::Experimental::Rotation3D+;
#pragma link C++ class ROOT::Experimental::AxisAngle+;
#pragma link C++ class ROOT::Experimental::EulerAngles+;
#pragma link C++ class ROOT::Experimental::Quaternion+;
#pragma link C++ class ROOT::Experimental::RotationZYX+;
#pragma link C++ class ROOT::Experimental::RotationX+;
#pragma link C++ class ROOT::Experimental::RotationY+;
#pragma link C++ class ROOT::Experimental::RotationZ+;
#pragma link C++ class ROOT::Experimental::LorentzRotation+;
#pragma link C++ class ROOT::Experimental::Boost+;
#pragma link C++ class ROOT::Experimental::BoostX+;
#pragma link C++ class ROOT::Experimental::BoostY+;
#pragma link C++ class ROOT::Experimental::BoostZ+;


#pragma link C++ class ROOT::Experimental::Plane3D+;
#pragma link C++ class ROOT::Experimental::Transform3D+;
#pragma link C++ class ROOT::Experimental::Translation3D+;

//#endif

// typedef's


#pragma link C++ typedef ROOT::Experimental::XYVector;
#pragma link C++ typedef ROOT::Experimental::Polar2DVector;

#pragma link C++ typedef ROOT::Experimental::XYPoint;
#pragma link C++ typedef ROOT::Experimental::Polar2DPoint;

#pragma link C++ typedef ROOT::Experimental::XYZVector;
#pragma link C++ typedef ROOT::Experimental::RhoEtaPhiVector;
#pragma link C++ typedef ROOT::Experimental::Polar3DVector;

#pragma link C++ typedef ROOT::Experimental::XYZPoint;
#pragma link C++ typedef ROOT::Experimental::RhoEtaPhiPoint;
#pragma link C++ typedef ROOT::Experimental::Polar3DPoint;

#pragma link C++ typedef ROOT::Experimental::XYZTVector;
#pragma link C++ typedef ROOT::Experimental::PtEtaPhiEVector;
#pragma link C++ typedef ROOT::Experimental::PxPyPzMVector;
#pragma link C++ typedef ROOT::Experimental::PtEtaPhiMVector;

#pragma link C++ typedef ROOT::Experimental::RhoZPhiVector;
#pragma link C++ typedef ROOT::Experimental::PxPyPzEVector;

// tyoedef for floating types

#pragma link C++ typedef ROOT::Experimental::XYVectorF;
#pragma link C++ typedef ROOT::Experimental::Polar2DVectorF;

#pragma link C++ typedef ROOT::Experimental::XYPointF;
#pragma link C++ typedef ROOT::Experimental::Polar2DPointF;

#pragma link C++ typedef ROOT::Experimental::XYZVectorF;
#pragma link C++ typedef ROOT::Experimental::RhoEtaPhiVectorF;
#pragma link C++ typedef ROOT::Experimental::Polar3DVectorF;

#pragma link C++ typedef ROOT::Experimental::XYZPointF;
#pragma link C++ typedef ROOT::Experimental::RhoEtaPhiPointF;
#pragma link C++ typedef ROOT::Experimental::Polar3DPointF;

#pragma link C++ typedef ROOT::Experimental::XYZTVectorF;

// dictionary for points and vectors functions
// not needed with Cling
//#include "LinkDef_Vector3D.h"
//#include "LinkDef_Point3D.h"
//#include "LinkDef_Vector4D.h"
//#include "LinkDef_Rotation.h"

// for std::vector of genvectors
#include "LinkDef_GenVector2.h"


// utility functions

#pragma link C++ namespace ROOT::Experimental::VectorUtil;



#endif
