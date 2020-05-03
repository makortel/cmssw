#ifndef MagneticField_VolumeBasedEngine_localVolumeBasedMagneticField_h
#define MagneticField_VolumeBasedEngine_localVolumeBasedMagneticField_h

#include "MagneticField/Engine/interface/localMagneticField.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

namespace local {
  class VolumeBasedMagneticField : private detail::MagneticFieldT<::VolumeBasedMagneticField> {
    using Base = detail::MagneticFieldT<::VolumeBasedMagneticField>;

    // For tests
    friend class ::testMagneticField;
    friend class ::testMagGeometryAnalyzer;

  public:
    VolumeBasedMagneticField() = default;
    explicit VolumeBasedMagneticField(const local::MagneticField& field) : Base(field) {}

    using Base::inInverseGeV;
    using Base::inKGauss;
    using Base::inTesla;
    using Base::inTeslaUnchecked;
    using Base::isDefined;
    using Base::nominalValue;

    bool isValid() const { return field() != nullptr; }

    const MagVolume* findVolume(const GlobalPoint& gp) { return field()->findVolume(gp, cache()); }
  };
}  // namespace local

#endif
