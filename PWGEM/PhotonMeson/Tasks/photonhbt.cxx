// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file photonhbt.cxx
/// \brief This tasks primary role is to generate the 2D and 3D photonHBT functions with different mixing techniques.
/// \author Daiki Sekihata, daiki.sekihata@cern.ch
///         Stefanie Mrozinski, stefanie.mrozinski@cern.ch

#include "PWGEM/Dilepton/Utils/EventMixingHandler.h"
#include "PWGEM/PhotonMeson/Core/EMPhotonEventCut.h"
#include "PWGEM/PhotonMeson/Core/V0PhotonCut.h"
#include "PWGEM/PhotonMeson/DataModel/EventTables.h"
#include "PWGEM/PhotonMeson/DataModel/gammaTables.h"
#include "PWGEM/PhotonMeson/Utils/AnalyticV0.h"
#include "PWGEM/PhotonMeson/Utils/EventHistograms.h"
#include "PWGEM/PhotonMeson/Utils/MCUtilities.h"
#include "PWGEM/PhotonMeson/Utils/PairUtilities.h"

#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h" // getTrackParCov
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/EventSelection.h"

#include <CCDB/BasicCCDBManager.h>
#include <CommonConstants/MathConstants.h>
#include <CommonConstants/PhysicsConstants.h>
#include <DataFormatsParameters/GRPMagField.h>
#include <DataFormatsTPC/VDriftCorrFact.h>
#include <Framework/ASoA.h>
#include <Framework/ASoAHelpers.h>
#include <Framework/AnalysisDataModel.h>
#include <Framework/AnalysisHelpers.h>
#include <Framework/AnalysisTask.h>
#include <Framework/Concepts.h>
#include <Framework/Configurable.h>
#include <Framework/HistogramRegistry.h>
#include <Framework/HistogramSpec.h>
#include <Framework/InitContext.h>
#include <Framework/Logger.h>
#include <Framework/OutputObjHeader.h>
#include <Framework/runDataProcessing.h>
#include <MathUtils/Utils.h>
#include <ReconstructionDataFormats/Track.h>

#include <Math/GenVector/Boost.h>
#include <Math/Vector3D.h> // IWYU pragma: keep
#include <Math/Vector3Dfwd.h>
#include <Math/Vector4D.h> // IWYU pragma: keep
#include <Math/Vector4Dfwd.h>
#include <TH1.h>
#include <TPDGCode.h>
#include <TString.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace o2;
using namespace o2::aod;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace o2::soa;
using namespace o2::aod::pwgem::dilepton::utils;
namespace pairutil = o2::aod::pwgem::photonmeson::utils::pairutil;
namespace pmmc = o2::aod::pwgem::photonmeson::utils::mcutil;
using namespace o2::pwgem::photonmeson;

// ─── Event Information Tables ────────────────────────────────────────────

using MyCollisions = soa::Join<aod::PMEvents, aod::EMEventsAlias, aod::EMEventsMult_000, aod::EMEventsCent_000, aod::EMEventsQvec_001>;
using MyCollisionsMC = soa::Join<aod::PMEvents, aod::EMEventsAlias, aod::EMEventsMult_000, aod::EMEventsCent_000, aod::EMEventsQvec_001, aod::EMMCEventLabels>;
using MyCollision = MyCollisions::iterator;

// ─── Photon Tables ────────────────────────────────────────────

using MyV0Photons = soa::Join<aod::V0PhotonsKF, aod::V0KFEMEventIds, aod::V0PhotonsPhiVPsi>;
using MyV0Photon = MyV0Photons::iterator;

// ─── Assoc. Track Tables ────────────────────────────────────────────

using MyMCV0Legs = soa::Join<aod::V0Legs, aod::V0LegMCLabels>;
using MyV0LegsXYZ = soa::Join<aod::V0Legs, aod::V0LegsXYZ>;
using MyMCV0LegsXYZ = soa::Join<aod::V0Legs, aod::V0LegMCLabels, aod::V0LegsXYZ>;

using MyTracksIU = soa::Join<aod::TracksIU, aod::TracksCovIU>;
using MyMCV0Leg = MyMCV0Legs::iterator;

static constexpr float kMinMagnitude = 1e-12f;
static constexpr float kMinCosine = 1e-12f;
static constexpr float kMinSigma = 1e-9f;

static constexpr std::array<float, 17> kPhiStarRadiiM = {
  0.85f, 0.95f, 1.05f, 1.15f, 1.25f, 1.35f, 1.45f, 1.55f, 1.65f,
  1.75f, 1.85f, 1.95f, 2.05f, 2.15f, 2.25f, 2.35f, 2.45f};
static constexpr float kPhiStarInvalid = -999.f;
static constexpr float kCmPerM = 100.f;

struct LegHelix {
  float phiStar = kPhiStarInvalid;
  float sxy = -1.f;
  int status = 3;
};

[[nodiscard]] static LegHelix legHelixAt(float vxM, float vyM, float phiMom, float pt, int charge, float bzT, float radiusM)
{
  LegHelix out;
  if (pt < kMinSigma || std::fabs(bzT) < kMinSigma) {
    out.status = 3; // o2-linter: disable=magic-number (status code)
    return out;
  }
  const float rStart = std::hypot(vxM, vyM);
  if (rStart < kMinSigma) {
    out.status = 3; // o2-linter: disable=magic-number (status code)
    return out;
  }
  if (radiusM < rStart) {
    out.status = 1;
    return out;
  }
  const float rho = pt / (0.3f * std::fabs(bzT));
  const float sgn = (bzT >= 0.f ? 1.f : -1.f) * static_cast<float>(charge);
  const float cx = vxM + sgn * rho * std::sin(phiMom);
  const float cy = vyM - sgn * rho * std::cos(phiMom);
  const float d = std::hypot(cx, cy);
  if (d < kMinSigma) {
    out.status = 3; // o2-linter: disable=magic-number (status code)
    return out;
  }
  if (radiusM > d + rho || radiusM < std::fabs(d - rho)) {
    out.status = 2; // o2-linter: disable=magic-number (status code)
    return out;
  }
  const float phiC = std::atan2(cy, cx);
  auto clamp1 = [](float x) { return std::max(-1.f, std::min(1.f, x)); };
  auto alphaAt = [&](float r) { return std::acos(clamp1((r * r + d * d - rho * rho) / (2.f * r * d))); };
  auto betaAt = [&](float r) { return std::acos(clamp1((d * d + rho * rho - r * r) / (2.f * d * rho))); };

  const float phiVtx = std::atan2(vyM, vxM);
  const float a0 = alphaAt(rStart);
  const float resPlus = std::fabs(RecoDecay::constrainAngle(phiC + a0 - phiVtx, -o2::constants::math::PI));
  const float resMinus = std::fabs(RecoDecay::constrainAngle(phiC - a0 - phiVtx, -o2::constants::math::PI));
  const float branch = (resPlus < resMinus) ? 1.f : -1.f;

  out.phiStar = RecoDecay::constrainAngle(phiC + branch * alphaAt(radiusM), 0.f);
  out.sxy = rho * std::fabs(betaAt(radiusM) - betaAt(rStart));
  out.status = 0;
  return out;
}

[[nodiscard]] inline float selfTestLegHelix()
{
  float worst = 0.f;
  constexpr float eps = 1.e-5f;

  for (const auto& bz : {0.5f, -0.5f}) {
    for (const auto& q : {+1, -1}) {
      for (const auto& pt : {0.1f, 0.2f, 0.5f}) {
        for (const auto& r : {0.85f, 1.45f, 2.45f}) {
          for (const auto& phi : {0.f, 1.f, -2.f}) {
            const auto h = legHelixAt(eps * std::cos(phi), eps * std::sin(phi), phi, pt, q, bz, r);

            if (h.status != 0) {
              continue;
            }
            const float arg =
              -0.3f * bz * static_cast<float>(q) * r / (2.f * pt);

            if (std::fabs(arg) >= 1.f) {
              continue;
            }
            const float referencePhi =
              RecoDecay::constrainAngle(phi + std::asin(arg), 0.f);

            worst = std::max(
              worst,
              std::fabs(RecoDecay::constrainAngle(
                h.phiStar - referencePhi,
                -o2::constants::math::PI)));
          }
        }
      }
    }
  }

  return worst;
}

struct Photonhbt {

  struct PairQAObservables {
    ROOT::Math::PtEtaPhiMVector v1, v2, k12;
    float x1 = 0.f, y1 = 0.f, z1 = 0.f, x2 = 0.f, y2 = 0.f, z2 = 0.f;
    float r1 = 0.f, r2 = 0.f, dx = 0.f, dy = 0.f, dz = 0.f;
    float deltaR = 0.f, deltaZ = 0.f, deltaRxy = 0.f, deltaR3D = 0.f;
    float opa = 0.f, cosOA = 0.f, drOverCosOA = 0.f;
    float deta = 0.f, dphi = 0.f, pairEta = 0.f, pairPhi = 0.f;
    float kt = 0.f, qinv = 0.f, cosTheta = 0.f, openingAngle = 0.f;
    bool valid = true;
  };

  struct PairSep {
    float dMin3D{999.f}, dRPhiAtMin3D{999.f}, dZAtMin3D{999.f};
    float dMinRPhi{999.f}, dZAtMinRPhi{999.f};
    std::array<float, 4> fCloseRPhiThr{};
    std::array<int, 4> nCloseRPhiThr{};
    float fCloseRPhi{0.f};
    int nCommon{0};
    int nSameSide{0};
    int criticalCharge{-1}; // 0 = e+e+, 1 = e-e-, chosen by d_3D
    int rphiCharge{-1};
    std::array<float, 2> dMin3DPerCharge{999.f, 999.f};
    std::array<float, 2> fCloseRPhiPerCharge{0.f, 0.f};
    std::array<float, 2> dMinRPhiPerCharge{999.f, 999.f};
    std::array<int, 2> nSameSidePerCharge{};
    std::array<std::array<int, 4>, 2> nCloseRPhiThrPerCharge{};
    std::array<std::array<float, 4>, 2> fCloseRPhiThrPerCharge{};
    float dMinRPhiAll{999.f}, dPairMinAll{999.f};
    float dPairMin{999.f};
    float dRPhiAtDPairMin{999.f}, dZAtDPairMin{999.f};
    float rAtDPairMin{-1.f}, absZAtDPairMin{-1.f}, driftLen{-1.f};
    int nCloseScaled{0};
    std::array<float, 2> dPairMinPerCharge{999.f, 999.f};
    float dZGlobalAtDPairMin{999.f};
    float dZSignedLocalAtDPairMin{999.f}, dZSignedGlobalAtDPairMin{999.f};
    float dVtxZ{0.f};
    float fCloseScaled{0.f};
    float driftOldAtDPairMin{-1.f};
    bool sameSideAtDPairMin{true};
    int nLegsITSTPC{0};
    std::array<int, 2> nPoints{};
    std::array<std::array<float, kPhiStarRadiiM.size()>, 2> ptDRPhi{}, ptDZ{}, ptDZSgnLocal{}, ptR{}, ptDrift{}, ptSameSide{};
  };

  struct CrossObs {
    std::array<AltPairing, 2> alt{};
    int nAltValid{0};
    float maxAltDca{999.f};
    float maxAltDcaXY{999.f};
    float maxAltDcaZ{999.f};
    std::array<float, 2> mee{999.f, 999.f};
    float meeOverQ{999.f};
    float ownMeeMax{999.f};
    float deltaMee{-999.f};
    std::array<float, 2> ownMee{999.f, 999.f};
    float m4{999.f};
  };

  struct TruthGamma {
    int id = -1, posId = -1, negId = -1;
    float eta = 0.f, phi = 0.f, pt = 0.f;
    float rTrue = -1.f;
    float vxTrue = 0.f, vyTrue = 0.f, vzTrue = 0.f;
    float legDRtrue = -1.f;
    float legDEta = 0.f;
    float legDPhi = 0.f;
    float alphaTrue = 0.f;
    std::array<float, 2> legPtTrue{}, legEtaTrue{}, legPhiTrue{};
    bool legsInV0 = false;
    bool v0Built = false;
    bool v0Selected = false;
  };

  struct PhotonWithLegs {
    float fPt{0}, fEta{0}, fPhi{0};
    float fVx{0}, fVy{0}, fVz{0};
    float fVtxZ{0};
    float fVtxX{0}, fVtxY{0};
    float fDcaXYToPV{0}, fDcaZToPV{0};
    std::array<float, 2> fLegDcaZ{};
    std::array<float, 2> fLegPt{}, fLegEta{}, fLegPhi{};
    std::array<float, 2> fLegNClsFindable{}, fLegNClsITS{};
    std::array<std::array<float, kPhiStarRadiiM.size()>, 2> fLegPhiStar{};
    std::array<std::array<float, kPhiStarRadiiM.size()>, 2> fLegSxy{}; // transverse arc length since conversion (m)
    [[nodiscard]] float pt() const { return fPt; }
    [[nodiscard]] float eta() const { return fEta; }
    [[nodiscard]] float phi() const { return fPhi; }
    [[nodiscard]] float vx() const { return fVx; }
    [[nodiscard]] float vy() const { return fVy; }
    [[nodiscard]] float vz() const { return fVz; }
    [[nodiscard]] float legEta(int i) const { return fLegEta[i]; }
    [[nodiscard]] float legPhi(int i) const { return fLegPhi[i]; }
    [[nodiscard]] float legPt(int i) const { return fLegPt[i]; }
    std::array<float, 2> fLegNsigEl{}; // TPC electron nsigma per leg, for the mixed-event leg similarity
    [[nodiscard]] float legNsigEl(int i) const { return fLegNsigEl[i]; }
    int fNITSTPC{0};
    int fNForeignLegs{0};
    int fNTimedLegs{0};
    pairutil::V0PhotonLegCounts fLegCounts{};
    [[nodiscard]] pairutil::V0PhotonLegCounts const& legCounts() const { return fLegCounts; }
    int fIsTruePhoton{-1};
    int64_t fGlobalIndex{-1};
    std::array<int64_t, 2> fLegTrackId{{-1, -1}}; // reconstructed track of each leg: two V0s sharing a track are never paired
    std::array<uint8_t, 2> fLegDetMask{{0, 0}};   // per leg: bit0 ITS, bit1 TPC, bit2 TRD, bit3 TOF (to recompute V0PhotonLegCounts for a cross candidate)
    std::array<int8_t, 2> fLegEvent{{0, 0}};      // per leg: 0 = this event, 1 = the pool event
    float fAnaScore{-1.f};
    float fAnaPca{999.f};
    float fAnaCosPA{-2.f};
    std::array<float, 2> fLegZ0{{0.f, 0.f}};
    std::array<o2::track::TrackParCov, 2> fLegTrack{};
    bool fHasTrackCov{false};
    std::array<std::array<float, 3>, 2> fLegXYZ{};
    bool fHasLegXYZ{false};
    float fAnaDcaXY{999.f};
    float fAnaDcaZ{999.f};
    bool fAnaPhotonLike{false};
    [[nodiscard]] bool sharesTrackWith(PhotonWithLegs const& o) const
    {
      for (int i = 0; i < 2; ++i) {   // o2-linter: disable=magic-number (two legs)
        for (int j = 0; j < 2; ++j) { // o2-linter: disable=magic-number (two legs)
          if (fLegTrackId[i] == o.fLegTrackId[j] && fLegEvent[i] == o.fLegEvent[j]) {
            return true;
          }
        }
      }
      return false;
    }
  };

  struct CrossPhotonLite {
    float fPt{0}, fEta{0}, fPhi{0};
    float fVx{0}, fVy{0}, fVz{0};
    float fDcaXYToPV{999.f}, fDcaZToPV{999.f};
    [[nodiscard]] float pt() const { return fPt; }
    [[nodiscard]] float eta() const { return fEta; }
    [[nodiscard]] float phi() const { return fPhi; }
    [[nodiscard]] float vx() const { return fVx; }
    [[nodiscard]] float vy() const { return fVy; }
    [[nodiscard]] float vz() const { return fVz; }
  };

  struct Emulation {
    int outcome{0}; // 0 = not evaluated
    bool swapped{false};
    float sStoredBest{999.f};
    float sCrossBest{999.f};
    float dSBest{999.f};
    bool crossExists{false};
    std::array<int, 2> failMask{{0, 0}};
    AnalyticV0 x1{}, x2{};
  };

  struct PhotonMCInfo {
    bool hasMC = false;
    bool sameMother = false;
    bool isTruePhoton = false;
    int mcPosId = -1;
    int mcNegId = -1;
    int motherId = -1;
    int motherPdg = 0;
    bool isPhysicalPrimary = false;
    int posMotherId = -1;
    int negMotherId = -1;
    bool posMotherIsPhoton = false;
    bool negMotherIsPhoton = false;
  };

  struct DedupCand {
    int64_t gi{-1};
    float eta{0.f}, phi{0.f}, pt{0.f}, rConv{0.f}, chi2{1e9f};
    float vx{0.f}, vy{0.f}, vz{0.f};
    int nITSTPC{0};

    // Legs, Index 0 = e+, 1 = e-
    std::array<float, 2> legPt{}, legEta{}, legPhi{}, legDeDx{}, legFracShared{};
    std::array<bool, 2> legHasTpc{false, false};
    std::array<int64_t, 2> legTrackId{{-1, -1}};

    PhotonMCInfo mc{};
  };

  struct LegDistance {
    float dEta{0.f}, dPhi{0.f}, ptAsym{1.f}, dedxAsym{1.f}, fShared{0.f};
    bool dedxValid{false};
  };

  enum class PairTruthType : uint8_t {
    Unknown = 0,
    TrueTrueDistinct,
    TrueTrueSamePhoton,
    SharedMcLeg,
    TrueFake,
    FakeFake,
    Pi0Daughters,
  };

  /*************************************************/
  // CONFIGURABLES
  /*************************************************/

  Service<o2::ccdb::BasicCCDBManager> ccdb{};
  Configurable<std::string> cfgCcdbUrl{"cfgCcdbUrl", "http://alice-ccdb.cern.ch", "CCDB url"};
  Configurable<float> cfgBzOverrideT{"cfgBzOverrideT", -999.f, "Bz in Tesla; used instead of CCDB if > -100"};
  float mBzT{0.f};           // signed, Tesla
  float mVDriftCmPerNs{0.f}; // TPC drift velocity from CCDB; 0 = unavailable

  struct : ConfigurableGroup {
    std::string prefix = "qaflags_group";
    Configurable<float> cfgMaxQinvForProcessing{"cfgMaxQinvForProcessing", 0.5, "skip mixed pairs with q_inv above this before building observables"};
  } qaflags;

  // ─── HBT analysis mode ───────────────────────────────────────────────────────────
  struct : ConfigurableGroup {
    std::string prefix = "hbtanalysis_group";
    Configurable<bool> cfgDo3D{"cfgDo3D", false, "enable 3D (qout,qside,qlong) analysis"};
    Configurable<bool> cfgDo2D{"cfgDo2D", false, "enable 2D (qout,qinv) projection (requires cfgDo3D)"};
    Configurable<bool> cfgDo2DSideLong{"cfgDo2DSideLong", false, "additionally book/fill CF_2D_Side and CF_2D_Long (qside/qlong vs qinv; two extra 5D sparses -> costs fill time and merge size; requires cfgDo2D)"};
    Configurable<bool> cfgUseLCMS{"cfgUseLCMS", false, "measure 1D relative momentum in LCMS"};
    Configurable<bool> cfgDoQinvGate3D{"cfgDoQinvGate3D", false, "book/fill CF_3D_Qinv: 3D LCMS CF with a COARSE qinv axis (edges = candidate gate values)"};
    ConfigurableAxis confMultNTracksBins{"confMultNTracksBins", {VARIABLE_WIDTH, 0., 100., 200., 300., 400., 500., 600, 700, 800, 900, 1000., 1100., 1200., 1300., 1400., 1500., 1600., 1700., 1800., 1900., 2000., 2100., 2200., 2300., 2400., 2500., 2600., 2700., 2800., 2900., 3000., 4000., 5000.}, "N_{tracks}^{PV} |#eta|<1 bins for CF sparses"};
    ConfigurableAxis confMultFT0MBins{"confMultFT0MBins", {VARIABLE_WIDTH, 0., 25., 50., 100., 200., 400., 800., 2000., 2500., 3000., 3500., 4000., 5000., 6000., 7000., 8000., 9000., 10000., 15000., 40000., 100000., 250000.}, "FT0M amplitude bins for CF sparses"};
  } hbtanalysis;

  // ----- Photon Leg Classification

  Configurable<bool> cfgDoPhotonClassPairCut{"cfgDoPhotonClassPairCut", false, "apply photon-class pair selection A x B (leg track composition)"};

  struct : ConfigurableGroup {
    std::string prefix = "photonclassA_group";
    Configurable<int> cfgMinNLegsITSTPC{"cfgMinNLegsITSTPC", 0, "min number of ITS-TPC legs (0-2)"};
    Configurable<int> cfgMaxNLegsITSTPC{"cfgMaxNLegsITSTPC", 2, "max number of ITS-TPC legs (0-2)"};
    Configurable<int> cfgMinNLegsITSOnly{"cfgMinNLegsITSOnly", 0, "min number of ITS-only legs (0-2)"};
    Configurable<int> cfgMaxNLegsITSOnly{"cfgMaxNLegsITSOnly", 2, "max number of ITS-only legs (0-2)"};
    Configurable<int> cfgMinNLegsTPCOnly{"cfgMinNLegsTPCOnly", 0, "min number of TPC-only legs (0-2)"};
    Configurable<int> cfgMaxNLegsTPCOnly{"cfgMaxNLegsTPCOnly", 2, "max number of TPC-only legs (0-2)"};
    Configurable<int> cfgMinNLegsTRD{"cfgMinNLegsTRD", 0, "min number of legs with TRD (0-2)"};
    Configurable<int> cfgMaxNLegsTRD{"cfgMaxNLegsTRD", 2, "max number of legs with TRD (0-2)"};
    Configurable<int> cfgMinNLegsTOF{"cfgMinNLegsTOF", 0, "min number of legs with TOF (0-2)"};
    Configurable<int> cfgMaxNLegsTOF{"cfgMaxNLegsTOF", 2, "max number of legs with TOF (0-2)"};
  } photonclassA;

  struct : ConfigurableGroup {
    std::string prefix = "photonclassB_group";
    Configurable<int> cfgMinNLegsITSTPC{"cfgMinNLegsITSTPC", 0, "min number of ITS-TPC legs (0-2)"};
    Configurable<int> cfgMaxNLegsITSTPC{"cfgMaxNLegsITSTPC", 2, "max number of ITS-TPC legs (0-2)"};
    Configurable<int> cfgMinNLegsITSOnly{"cfgMinNLegsITSOnly", 0, "min number of ITS-only legs (0-2)"};
    Configurable<int> cfgMaxNLegsITSOnly{"cfgMaxNLegsITSOnly", 2, "max number of ITS-only legs (0-2)"};
    Configurable<int> cfgMinNLegsTPCOnly{"cfgMinNLegsTPCOnly", 0, "min number of TPC-only legs (0-2)"};
    Configurable<int> cfgMaxNLegsTPCOnly{"cfgMaxNLegsTPCOnly", 2, "max number of TPC-only legs (0-2)"};
    Configurable<int> cfgMinNLegsTRD{"cfgMinNLegsTRD", 0, "min number of legs with TRD (0-2)"};
    Configurable<int> cfgMaxNLegsTRD{"cfgMaxNLegsTRD", 2, "max number of legs with TRD (0-2)"};
    Configurable<int> cfgMinNLegsTOF{"cfgMinNLegsTOF", 0, "min number of legs with TOF (0-2)"};
    Configurable<int> cfgMaxNLegsTOF{"cfgMaxNLegsTOF", 2, "max number of legs with TOF (0-2)"};
  } photonclassB;

  // ─── Event mixing ─────────────────────────────────────────────────────────────
  struct : ConfigurableGroup {
    std::string prefix = "mixing_group";
    Configurable<bool> cfgDoMix{"cfgDoMix", true, "flag for event mixing"};
    Configurable<int> ndepth{"ndepth", 100, "depth for event mixing"};
    Configurable<uint64_t> ndiffBCMix{"ndiffBCMix", 594, "difference in global BC required for mixed events"};
    Configurable<int> cfgEP2EstimatorForMix{"cfgEP2EstimatorForMix", 3, "FT0M:0, FT0A:1, FT0C:2, FV0A:3, BTot:4, BPos:5, BNeg:6"};
    Configurable<int> cfgOccupancyEstimator{"cfgOccupancyEstimator", 0, "FT0C:0, Track:1"};
    Configurable<int> cfgCentEstimator{"cfgCentEstimator", 2, "FT0M:0, FT0A:1, FT0C:2"};
    ConfigurableAxis confVtxBins{"confVtxBins", {VARIABLE_WIDTH, -10.f, -8.f, -6.f, -4.f, -2.f, 0.f, 2.f, 4.f, 6.f, 8.f, 10.f}, "Mixing bins - z-vertex"};
    ConfigurableAxis confCentBins{"confCentBins", {VARIABLE_WIDTH, 0.f, 5.f, 10.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f, 90.f, 100.f, 999.f}, "Mixing bins - centrality"};
    ConfigurableAxis confEPBinsBins{"confEPBinsBins", {16, -o2::constants::math::PIHalf, +o2::constants::math::PIHalf}, "Mixing bins - EP angle"};
    ConfigurableAxis confOccupancyBins{"confOccupancyBins", {VARIABLE_WIDTH, -1, 1e+10}, "Mixing bins - occupancy"};
    Configurable<int> cfgMixMode{"cfgMixMode", 1, "0 = photon-level: the built photons of two events are paired as they are; 1 = leg-level: the four legs of every mixed pair go through the builder's candidate competition (a, b, x1, x2) and the survivors are paired; 2 = leg-built reference: forget the stored pairing, build e+(A)e-(B) and e+(B)e-(A) analytically (PCA, R window), one use per leg by smallest PCA, pair G_AB x G_BA - no stored photon in the denominator"};
    Configurable<float> cfgQuartetMaxQinv{"cfgQuartetMaxQinv", 0.15f, "mixed pairs with q_inv(a, b) below this go through the four-leg candidate competition; above, four legs are never collinear enough for a cross candidate"};
    Configurable<int> cfgCrossBuilder{"cfgCrossBuilder", 0, "cfgMixMode 1, skim and V0LegsXYZ paths: how a candidate is built from two legs: 0 = analytic helix circles, 1 = DCAFitter on placeholder covariances"};
    Configurable<float> cfgScoreWeight{"cfgScoreWeight", 0.5f, "builder score weight w: w*cosPA-term + (1-w)*pca-term (getScoreV0), for the stored V0s and the cross candidates alike"};
  } mixing;

  struct : ConfigurableGroup {
    std::string prefix = "mixbuilder_group";
    Configurable<float> cfgBuilderMinCosPA{"cfgBuilderMinCosPA", 0.99f, "cfgMixMode 1: builder-level acceptance of a cross candidate (the V0 builder's own cuts, NOT the analysis cuts): min cosPA"};
    Configurable<float> cfgBuilderMaxPca{"cfgBuilderMaxPca", 1.5f, "cfgMixMode 1: builder-level acceptance: max PCA of the two legs (cm)"};
    Configurable<float> cfgBuilderMinR{"cfgBuilderMinR", 1.f, "cfgMixMode 1: builder-level acceptance: min conversion radius (cm)"};
    Configurable<float> cfgBuilderMaxR{"cfgBuilderMaxR", 180.f, "cfgMixMode 1: builder-level acceptance: max conversion radius (cm)"};
    Configurable<float> cfgBuilderMaxAlpha{"cfgBuilderMaxAlpha", 0.95f, "cfgMixMode 1: builder-level acceptance: max |alpha| (Armenteros)"};
    Configurable<float> cfgBuilderMaxQt{"cfgBuilderMaxQt", 0.05f, "cfgMixMode 1: builder-level acceptance: max qT (Armenteros, GeV/c)"};
  } mixbuilder;

  // ─── Centrality slection ─────────────────────────────────────────────────
  struct : ConfigurableGroup {
    std::string prefix = "centralitySelection_group";
    Configurable<float> cfgCentMin{"cfgCentMin", -1, "min. centrality"};
    Configurable<float> cfgCentMax{"cfgCentMax", 999, "max. centrality"};
  } centralitySelection;

  struct : ConfigurableGroup {
    std::string prefix = "mctruth_group";
    Configurable<float> cfgMCMaxQinv{"cfgMCMaxQinv", 0.3f, "upper q_inv for ALL truth-level pair loops (truth CF, PairEff): pairs above are skipped before any boost or fill - keep it >= the upper edge of confQBins"};
    Configurable<bool> cfgDoTruthMix{"cfgDoTruthMix", false, "fill the truth-level CF (MC/TruthCF/) with its own mixed event"};
    Configurable<int> cfgTruthMixDepth{"cfgTruthMixDepth", 10, "depth of the truth-level mixing pool"};
    Configurable<bool> cfgDoTruthLcms{"cfgDoTruthLcms", false, "fill the truth-level (q_out, q_inv, k_T) sparses for the truth CF, so a q_out window can be applied at truth level exactly as on the reconstructed CF"};
    Configurable<bool> cfgDoTruth3D{"cfgDoTruth3D", false, "additionally fill the full truth-level 3D LCMS sparse (q_out, q_side, q_long, k_T) - the truth analogue of CF_3D; off by default because it costs sparse occupancy"};
    Configurable<bool> cfgDoPairEff{"cfgDoPairEff", false, "fill the pair-efficiency sparses (MC/PairEff/) for same AND mixed truth pairs, so the non-factorising part of the efficiency can be separated from the part that cancels in the correlation function; needs cfgDoTruthMix for the mixed part"};
    Configurable<float> cfgMCMinV0Pt{"cfgMCMinV0Pt", 0.1f,
                                     "min pT for true photons in truth-efficiency loop (GeV/c); "
                                     "0 = fall back to pcmcuts.cfgMinPtV0"};
    Configurable<float> cfgMCMinLegPt{"cfgMCMinLegPt", 0.0f, "min pT for true e^{+}/e^{-} legs in truth-efficiency loop (GeV/c);"};
    Configurable<float> cfgMCMaxLegEta{"cfgMCMaxLegEta", 999.f, "max |eta| for true e^{+}/e^{-} legs in the truth list; 999 = no cut. Without pT and eta cuts the 'converted' denominator is dominated by soft photons whose legs can never be tracked"};
    Configurable<float> cfgMCMinRconv{"cfgMCMinRconv", 0.f, "min true conversion radius for photons in the truth list (cm); 0 = no cut"};
    Configurable<float> cfgMCMaxRconv{"cfgMCMaxRconv", 999.f, "max true conversion radius for photons in the truth list (cm); 999 = no cut"};
  } mctruth;

  struct : ConfigurableGroup {
    std::string prefix = "ggpaircut_group";
    Configurable<float> cfgMinDRCosOA{"cfgMinDRCosOA", -1.f, "min. dr/cosOA; <0 = disabled"};
    Configurable<bool> cfgDoRCut{"cfgDoRCut", false, "apply |R1-R2| > cfgMinDeltaR cut"};
    Configurable<float> cfgMinDeltaR{"cfgMinDeltaR", 0.f, "minimum |R1-R2| (cm)"};
    Configurable<bool> cfgDoZCut{"cfgDoZCut", false, "apply |DeltaZ| > cfgMinDeltaZ cut"};
    Configurable<float> cfgMinDeltaZ{"cfgMinDeltaZ", 0.f, "minimum |DeltaZ| (cm)"};
    Configurable<bool> cfgDoEllipseCut{"cfgDoEllipseCut", false, "reject pairs inside ellipse in DeltaEta-DeltaPhi"};
    Configurable<float> cfgEllipseSigEta{"cfgEllipseSigEta", 0.1f, "sigma_eta for ellipse cut"};
    Configurable<float> cfgEllipseSigPhi{"cfgEllipseSigPhi", 0.1f, "sigma_phi for ellipse cut"};
    Configurable<float> cfgEllipseR2{"cfgEllipseR2", 1.0f, "R^2 threshold: reject if ellipse value < R^2"};
    Configurable<float> cfgMaxAsymmetry{"cfgMaxAsymmetry", -1.f, "max |p_{T, 1} - p_{T, 2}|/(p_{T, 1} + p_{T, 2}) asymmetry cut"};
    Configurable<float> cfgMaxDcaZToPV{"cfgMaxDcaZToPV", 999.f, "max |DCAz| of both V0 photons to the primary vertex (cm); 999 = off"};
    Configurable<float> cfgMaxDcaXYToPV{"cfgMaxDcaXYToPV", 999.f, "max |DCAxy| of both V0 photons to the primary vertex (cm); 999 = off"};
  } ggpaircuts;

  struct : ConfigurableGroup {
    std::string prefix = "pairsep_group";
    Configurable<float> cfgSigRPhi{"cfgSigRPhi", 1.f, "rphi scale of the pair-merging metric (cm)"};
    Configurable<float> cfgSigZ{"cfgSigZ", 1.f, "z scale of the pair-merging metric (cm)"};
    Configurable<float> cfgSigZNs{"cfgSigZNs", -1.f, "z scale in ns of drift time; <=0 keeps cfgSigZ in cm"};
    Configurable<float> cfgDPairClose{"cfgDPairClose", 1.f, "a sampled radius counts as close when D_pair < this"};
    Configurable<std::vector<float>> cfgCloseThrCm{"cfgCloseThrCm", {0.5f, 1.0f, 1.5f, 2.0f}, "four d_rphi thresholds (cm) for N_close/f_close"};
    Configurable<bool> cfgDoPairMergeCut{"cfgDoPairMergeCut", false, "apply the pair-merging cut to SE AND ME"};
    Configurable<float> cfgDPairCut{"cfgDPairCut", 1.f, "reject pairs with min D_pair below this (needs cfgDoPairMergeCut)"};
    Configurable<int> cfgNCloseCut{"cfgNCloseCut", 0, "reject pairs with at least this many close radii; 0 disables"};
    Configurable<float> cfgTpcHalfLengthCm{"cfgTpcHalfLengthCm", 250.f, "TPC half length, used for drift length = halfLength - |z|"};
  } pairsep;

  struct : ConfigurableGroup {
    std::string prefix = "dedup_group";
    Configurable<bool> cfgDoDedup{"cfgDoDedup", false, "REMOVE duplicate-like candidates before pairing (the candidate-level duplicate QA lives in pairQCTask)"};
    Configurable<float> cfgDupMaxLegDEta{"cfgDupMaxLegDEta", 0.01f, "leg identical if |dEta| below this"};
    Configurable<float> cfgDupMaxLegDPhi{"cfgDupMaxLegDPhi", 0.01f, "leg identical if |dPhi| below this (rad)"};
    Configurable<float> cfgDupMaxLegPtAsym{"cfgDupMaxLegPtAsym", 0.05f, "leg identical if |pt1-pt2|/(pt1+pt2) below this"};
    Configurable<float> cfgDupMaxLegDeDxAsym{"cfgDupMaxLegDeDxAsym", 0.05f, "leg identical if |dEdx1-dEdx2|/(dEdx1+dEdx2) below this; skipped if a leg has no TPC"};
    Configurable<bool> cfgDupRequireBothLegs{"cfgDupRequireBothLegs", false, "true: both same-charge leg pairs must be identical; false: one is enough (also catches half-duplicates)"};
    Configurable<float> cfgDupMaxDVtx3D{"cfgDupMaxDVtx3D", 999.f, "additionally require the 3D conversion-point distance below this (cm); 999 = off"};
  } dedup;

  struct : ConfigurableGroup {
    std::string prefix = "crosspair_group";
    Configurable<bool> cfgDoCrossPairCut{"cfgDoCrossPairCut", false, "reject pairs with a photon-like crossed combination; applied to SE AND ME"};
    Configurable<float> cfgCrossMaxMeeRatio{"cfgCrossMaxMeeRatio", 0.2f, "veto mode 0: min(m_ee^cross)/q_inv below this; genuine pairs sit near 0.5, leg swaps near 0"};
    Configurable<int> cfgCrossVetoMode{"cfgCrossVetoMode", 1, "veto criterion when cfgDoCrossPairCut: 0 = m_ee ratio (old); 1 = geometric, the alternative pairings form photon-like vertices by the full cfgAlt* AND-chain; 2 = direct cut on maxAltDca alone (cfgCrossMinMaxAltDca); 3 = 1 OR 2, a pair is rejected if either criterion flags it"};
    Configurable<float> cfgCrossMinMaxAltDca{"cfgCrossMinMaxAltDca", 4.9f, "veto mode 2/3: reject the pair when maxAltDca is BELOW this (cm); the scan optimum was ~4.9 cm. 0 disables this criterion"};
    Configurable<float> cfgAltMaxQinv{"cfgAltMaxQinv", 0.15f, "evaluate the alternative pairings only below this q_inv; above, four legs are never collinear enough (counts as 0 valid alternatives)"};
    Configurable<float> cfgAltMaxDca{"cfgAltMaxDca", 1.5f, "an alternative pairing is photon-like if its two legs approach closer than this at their PCA (cm) ..."};
    Configurable<float> cfgAltMinR{"cfgAltMinR", 1.f, "... and the PCA lies at a radius above this (cm) ..."};
    Configurable<float> cfgAltMaxR{"cfgAltMaxR", 90.f, "... and below this (cm) ..."};
    Configurable<float> cfgAltMinCosPA{"cfgAltMinCosPA", 0.98f, "... and its summed momentum points back to the primary vertex with cosPA above this ..."};
    Configurable<float> cfgAltMaxDcaXY{"cfgAltMaxDcaXY", 3.f, "... and the line through the PCA along its summed momentum passes the primary vertex within this distance in xy (cm); the finder cuts at 1.4 cm, ~3 cm also flags swaps whose true pairings just failed it ..."};
    Configurable<float> cfgAltMaxDcaZ{"cfgAltMaxDcaZ", 999.f, "... and passes the primary vertex within this distance in z (cm); 999 = not used. No bending in z: expected to add little against swaps, kept as a control"};
    Configurable<float> cfgAltMaxMee{"cfgAltMaxMee", 0.1f, "... and its m_ee is below this (GeV/c^2)"};
    Configurable<bool> cfgAltRequireBoth{"cfgAltRequireBoth", true, "true: veto only if BOTH alternative pairings are photon-like (the leg-swap signature); false: one is enough"};
  } crosspair;

  struct : ConfigurableGroup {
    std::string prefix = "eventcut_group";
    Configurable<float> cfgZvtxMin{"cfgZvtxMin", -10.f, "min. Zvtx"};
    Configurable<float> cfgZvtxMax{"cfgZvtxMax", +10.f, "max. Zvtx"};
    Configurable<bool> cfgRequireSel8{"cfgRequireSel8", true, "require sel8"};
    Configurable<bool> cfgRequireFT0AND{"cfgRequireFT0AND", true, "require FT0AND"};
    Configurable<bool> cfgRequireNoTFB{"cfgRequireNoTFB", true, "require no TF border"};
    Configurable<bool> cfgRequireNoITSROFB{"cfgRequireNoITSROFB", true, "require no ITS ROF border"};
    Configurable<bool> cfgRequireNoSameBunchPileup{"cfgRequireNoSameBunchPileup", false, "require no same bunch pileup"};
    Configurable<bool> cfgRequireVertexITSTPC{"cfgRequireVertexITSTPC", false, "require Vertex ITSTPC"};
    Configurable<bool> cfgRequireGoodZvtxFT0vsPV{"cfgRequireGoodZvtxFT0vsPV", false, "require good Zvtx FT0 vs PV"};
    Configurable<int> cfgTrackOccupancyMin{"cfgTrackOccupancyMin", -2, "min. track occupancy"};
    Configurable<int> cfgTrackOccupancyMax{"cfgTrackOccupancyMax", 1000000000, "max. track occupancy"};
    Configurable<float> cfgFT0COccupancyMin{"cfgFT0COccupancyMin", -2.f, "min. FT0C occupancy"};
    Configurable<float> cfgFT0COccupancyMax{"cfgFT0COccupancyMax", 1000000000.f, "max. FT0C occupancy"};
    Configurable<bool> cfgRequireNoCollInTimeRangeStandard{"cfgRequireNoCollInTimeRangeStandard", false, "no coll in time range std"};
    Configurable<bool> cfgRequireNoCollInTimeRangeStrict{"cfgRequireNoCollInTimeRangeStrict", false, "no coll in time range strict"};
    Configurable<bool> cfgRequireNoCollInITSROFStandard{"cfgRequireNoCollInITSROFStandard", false, "no coll in ITS ROF std"};
    Configurable<bool> cfgRequireNoCollInITSROFStrict{"cfgRequireNoCollInITSROFStrict", false, "no coll in ITS ROF strict"};
    Configurable<bool> cfgRequireNoHighMultCollInPrevRof{"cfgRequireNoHighMultCollInPrevRof", false, "no HM coll in prev ROF"};
    Configurable<bool> cfgRequireGoodITSLayer3{"cfgRequireGoodITSLayer3", false, "ITS layer 3 OK"};
    Configurable<bool> cfgRequireGoodITSLayer0123{"cfgRequireGoodITSLayer0123", false, "ITS layers 0-3 OK"};
    Configurable<bool> cfgRequireGoodITSLayersAll{"cfgRequireGoodITSLayersAll", false, "all ITS layers OK"};
  } eventcuts;

  struct : ConfigurableGroup {
    std::string prefix = "pcmcut_group";
    Configurable<bool> cfgRequireV0WithITSTPC{"cfgRequireV0WithITSTPC", false, "select V0s with ITS-TPC tracks"};
    Configurable<bool> cfgRequireV0WithITSOnly{"cfgRequireV0WithITSOnly", false, "select V0s with ITS-only tracks"};
    Configurable<bool> cfgRequireV0WithTPCOnly{"cfgRequireV0WithTPCOnly", false, "select V0s with TPC-only tracks"};
    Configurable<float> cfgMinPtV0{"cfgMinPtV0", 0.1, "min pT for V0 photons at PV"};
    Configurable<float> cfgMaxEtaV0{"cfgMaxEtaV0", 0.8, "max eta for V0 photons at PV"};
    Configurable<float> cfgMinV0Radius{"cfgMinV0Radius", 16.0, "min V0 radius"};
    Configurable<float> cfgMaxV0Radius{"cfgMaxV0Radius", 90.0, "max V0 radius"};
    Configurable<float> cfgMaxAlphaAP{"cfgMaxAlphaAP", 0.95, "max alpha for AP cut"};
    Configurable<float> cfgMaxQtAP{"cfgMaxQtAP", 0.01, "max qT for AP cut"};
    Configurable<float> cfgMinCosPA{"cfgMinCosPA", 0.997, "min V0 CosPA"};
    Configurable<float> cfgMaxPCA{"cfgMaxPCA", 3.0, "max distance between 2 legs"};
    Configurable<float> cfgMaxChi2KF{"cfgMaxChi2KF", 1e+10, "max chi2/ndf with KF"};
    Configurable<bool> cfgRejectV0OnITSIB{"cfgRejectV0OnITSIB", true, "reject V0s on ITSib"};
    Configurable<bool> cfgDisableITSOnlyTrack{"cfgDisableITSOnlyTrack", false, "disable ITS-only tracks"};
    Configurable<bool> cfgDisableTPCOnlyTrack{"cfgDisableTPCOnlyTrack", false, "disable TPC-only tracks"};
    Configurable<int> cfgMinNClusterTPC{"cfgMinNClusterTPC", 70, "min ncluster TPC"};
    Configurable<int> cfgMinNCrossedRows{"cfgMinNCrossedRows", 70, "min crossed rows"};
    Configurable<float> cfgMaxFracSharedClustersTPC{"cfgMaxFracSharedClustersTPC", 999.f, "max fraction of shared TPC clusters"};
    Configurable<float> cfgMaxChi2TPC{"cfgMaxChi2TPC", 4.0, "max chi2/NclsTPC"};
    Configurable<float> cfgMaxChi2ITS{"cfgMaxChi2ITS", 36.0, "max chi2/NclsITS"};
    Configurable<float> cfgMinTPCNsigmaEl{"cfgMinTPCNsigmaEl", -3.5, "min TPC nsigma electron"};
    Configurable<float> cfgMaxTPCNsigmaEl{"cfgMaxTPCNsigmaEl", +3.5, "max TPC nsigma electron"};
  } pcmcuts;

  ConfigurableAxis confQBins{"confQBins", {60, 0, +0.3f}, "q bins for output histograms"};
  ConfigurableAxis confKtBins{"confKtBins", {VARIABLE_WIDTH, 0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75}, "kT bins"};
  ConfigurableAxis confDeltaEtaBins{"confDeltaEtaBins", {360, -1.6f, +1.6f}, "Delta-eta bins"};
  ConfigurableAxis confDeltaPhiBins{"confDeltaPhiBins", {360, -o2::constants::math::PI, o2::constants::math::PI}, "Delta-phi bins (rad)"};

  /*************************************************/
  // AXIS SPECS
  /*************************************************/

  AxisSpec axisKt{confKtBins, "k_{T} (GeV/c)"};
  AxisSpec axisQinv{confQBins, "q_{inv} (GeV/c)"};
  AxisSpec axisQabsLcms{confQBins, "|#bf{q}|^{LCMS} (GeV/c)"};
  AxisSpec axisQout{confQBins, "q_{out} (GeV/c)"};
  AxisSpec axisQside{confQBins, "q_{side} (GeV/c)"};
  AxisSpec axisQlong{confQBins, "q_{long} (GeV/c)"};
  AxisSpec axisDeltaEta{confDeltaEtaBins, "#Delta#eta"};
  AxisSpec axisDeltaPhi{confDeltaPhiBins, "#Delta#phi (rad)"};
  [[nodiscard]] AxisSpec makeAxisMultNTracks() const { return AxisSpec{hbtanalysis.confMultNTracksBins, "N_{tracks}^{PV}, |#eta| < 1"}; }
  [[nodiscard]] AxisSpec makeAxisMultFT0M() const { return AxisSpec{hbtanalysis.confMultFT0MBins, "mult. FT0M (amplitude)"}; }

  /*************************************************/
  // MAIN SPECS
  /*************************************************/

  HistogramRegistry fRegistry{"output", {}, OutputObjHandlingPolicy::AnalysisObject, false, false};
  HistogramRegistry fRegistryCF{"cf", {}, OutputObjHandlingPolicy::AnalysisObject, false, false};
  HistogramRegistry fRegistryPairMC{"pairMC", {}, OutputObjHandlingPolicy::AnalysisObject, false, false};
  HistogramRegistry fRegistryTruthMC{"truthMC", {}, OutputObjHandlingPolicy::AnalysisObject, false, false};

  EMPhotonEventCut fEMEventCut;
  V0PhotonCut fV0PhotonCut;

  int mRunNumber{0};
  bool isMC = false;
  int ndf = 0;
  pairutil::V0PhotonClassSelection mPhotonClassSelA{};
  pairutil::V0PhotonClassSelection mPhotonClassSelB{};

  std::vector<float> ztxBinEdges;
  std::vector<float> centBinEdges;
  std::vector<float> epBinEgdes;
  std::vector<float> occBinEdges;

  using MyEMH = o2::aod::pwgem::dilepton::utils::EventMixingHandler<std::tuple<int, int, int, int>, std::pair<int, int>, PhotonWithLegs>;
  std::shared_ptr<MyEMH> emh1;
  std::shared_ptr<MyEMH> emh2;
  PhotonLikeCuts mAltCuts{};

  struct CandidateCuts {
    float minPt{0.f}, maxEta{0.9f}, minCosPA{0.997f}, maxPca{1.f}, minR{16.f}, maxR{90.f}, maxAlpha{0.95f}, maxQt{0.05f}; // o2-linter: disable=magic-number (defaults = pcmcut_group)
  } mCandCuts{};

  [[nodiscard]] int builderFailMask(AnalyticV0 const& v) const
  {
    if (!v.ok) {
      return 1;
    }
    int m = 0;
    m |= (v.cospa < mixbuilder.cfgBuilderMinCosPA.value) ? 2 : 0;                                          // o2-linter: disable=magic-number (bit mask)
    m |= (v.pca > mixbuilder.cfgBuilderMaxPca.value) ? 4 : 0;                                              // o2-linter: disable=magic-number (bit mask)
    m |= (v.rxy() < mixbuilder.cfgBuilderMinR.value || v.rxy() > mixbuilder.cfgBuilderMaxR.value) ? 8 : 0; // o2-linter: disable=magic-number (bit mask)
    float alpha = 2.f, qt = 999.f;                                                                         // o2-linter: disable=magic-number (initialised to fail)
    v.armenteros(alpha, qt);
    m |= (std::fabs(alpha) >= mixbuilder.cfgBuilderMaxAlpha.value) ? 16 : 0; // o2-linter: disable=magic-number (bit mask)
    m |= (qt >= mixbuilder.cfgBuilderMaxQt.value) ? 32 : 0;                  // o2-linter: disable=magic-number (bit mask)
    return m;
  }

  [[nodiscard]] bool passBuilderCuts(AnalyticV0 const& v) const
  {
    if (!v.ok || v.cospa < mixbuilder.cfgBuilderMinCosPA.value || v.pca > mixbuilder.cfgBuilderMaxPca.value) {
      return false;
    }
    const float r = v.rxy();
    if (r < mixbuilder.cfgBuilderMinR.value || r > mixbuilder.cfgBuilderMaxR.value) {
      return false;
    }
    float alpha = 2.f, qt = 999.f; // o2-linter: disable=magic-number (initialised to fail)
    v.armenteros(alpha, qt);
    return std::fabs(alpha) < mixbuilder.cfgBuilderMaxAlpha.value && qt < mixbuilder.cfgBuilderMaxQt.value;
  }

  [[nodiscard]] bool passCandidateCuts(AnalyticV0 const& v) const
  {
    if (!v.ok || v.pt() < mCandCuts.minPt || std::fabs(v.eta()) > mCandCuts.maxEta) {
      return false;
    }
    if (v.cospa < mCandCuts.minCosPA || v.pca > mCandCuts.maxPca) {
      return false;
    }
    const float r = v.rxy();
    if (r < mCandCuts.minR || r > mCandCuts.maxR) {
      return false;
    }
    float alpha = 2.f, qt = 999.f; // o2-linter: disable=magic-number (initialised to fail)
    v.armenteros(alpha, qt);
    return std::fabs(alpha) < mCandCuts.maxAlpha && qt < mCandCuts.maxQt;
  }
  std::map<std::pair<int, int>, uint64_t> mapMixedEventIdToGlobalBC;
  std::map<std::tuple<int, int, int, int>, std::deque<std::vector<TruthGamma>>> truthGammaPool;

  SliceCache cache;
  Preslice<MyV0Photons> perCollisionPCM = aod::v0photonkf::pmeventId;
  PresliceUnsorted<MyMCV0Legs> perCollisionV0Legs = aod::v0leg::collisionId;
  PresliceUnsorted<aod::EMMCParticles> perMCCollisionEMMCParts = aod::emmcparticle::emmceventId;

  Filter collisionFilterCentrality =
    (centralitySelection.cfgCentMin < o2::aod::cent::centFT0M && o2::aod::cent::centFT0M < centralitySelection.cfgCentMax) ||
    (centralitySelection.cfgCentMin < o2::aod::cent::centFT0A && o2::aod::cent::centFT0A < centralitySelection.cfgCentMax) ||
    (centralitySelection.cfgCentMin < o2::aod::cent::centFT0C && o2::aod::cent::centFT0C < centralitySelection.cfgCentMax);
  Filter collisionFilterOccupancyTrack =
    eventcuts.cfgTrackOccupancyMin <= o2::aod::evsel::trackOccupancyInTimeRange &&
    o2::aod::evsel::trackOccupancyInTimeRange < eventcuts.cfgTrackOccupancyMax;
  Filter collisionFilterOccupancyFT0c =
    eventcuts.cfgFT0COccupancyMin <= o2::aod::evsel::ft0cOccupancyInTimeRange &&
    o2::aod::evsel::ft0cOccupancyInTimeRange < eventcuts.cfgFT0COccupancyMax;

  using FilteredMyCollisions = soa::Filtered<MyCollisions>;
  using FilteredMyMCCollisions = soa::Filtered<MyCollisionsMC>;

  /*************************************************/
  // INITS
  /*************************************************/

  void init(InitContext& context)
  {
    isMC = context.mOptions.get<bool>("processMC") || context.mOptions.get<bool>("processMCAOD") || context.mOptions.get<bool>("processMCXYZ");
    {
      const int nOn = static_cast<int>(context.mOptions.get<bool>("processAnalysis")) + static_cast<int>(context.mOptions.get<bool>("processAnalysisAOD")) + static_cast<int>(context.mOptions.get<bool>("processAnalysisXYZ")) +
                      static_cast<int>(context.mOptions.get<bool>("processMC")) + static_cast<int>(context.mOptions.get<bool>("processMCAOD")) + static_cast<int>(context.mOptions.get<bool>("processMCXYZ"));
      if (nOn != 1) {
        LOGF(fatal, "exactly one of processAnalysis/AOD/XYZ or processMC/AOD/XYZ must be enabled, got %d", nOn);
      }
    }
    if (pairsep.cfgCloseThrCm.value.size() != 4) { // o2-linter: disable=magic-number (four thresholds)
      LOGF(fatal, "cfgCloseThrCm must contain exactly four thresholds, got %zu",
           pairsep.cfgCloseThrCm.value.size());
    }
    if (pairsep.cfgCloseThrCm.value.front() <= 0.f ||
        !std::is_sorted(pairsep.cfgCloseThrCm.value.begin(), pairsep.cfgCloseThrCm.value.end())) {
      LOGF(fatal, "cfgCloseThrCm must be positive and sorted in ascending order");
    }
    mRunNumber = 0;
    parseBins(mixing.confVtxBins, ztxBinEdges);
    parseBins(mixing.confCentBins, centBinEdges);
    parseBins(mixing.confEPBinsBins, epBinEgdes);
    parseBins(mixing.confOccupancyBins, occBinEdges);
    emh1 = std::make_shared<MyEMH>(mixing.ndepth);
    emh2 = std::make_shared<MyEMH>(mixing.ndepth);
    DefineEMEventCut();
    DefinePCMCut();
    mPhotonClassSelA = pairutil::buildV0PhotonClassSelection(photonclassA);
    mPhotonClassSelB = pairutil::buildV0PhotonClassSelection(photonclassB);
    addhistograms();
    ccdb->setURL(cfgCcdbUrl);
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();

    const float worst = selfTestLegHelix();
    constexpr float kSelfTestTol = 1.e-4f;
    if (worst < kSelfTestTol) {
      LOGF(info, "photonhbt: leg-helix origin-limit self test PASSED (max |dphi| = %.2e rad)", worst);
    } else {
      LOGF(fatal,
           "photonhbt: leg-helix origin-limit self test FAILED, max |dphi| = %.3e rad. "
           "The helix-centre sign convention or the branch choice is wrong; every "
           "leg-separation variable would be mirrored. Refusing to run.",
           worst);
    }
    LOGF(info, "photonhbt pair merging cut: %s, D_pair > %.2f, N_close < %d (0 = off)",
         pairsep.cfgDoPairMergeCut.value ? "ON" : "off", pairsep.cfgDPairCut.value, pairsep.cfgNCloseCut.value);
    LOGF(info, "photonhbt: crossed-pair cut %s, mode %d (0 = m_ee ratio > %.2f; 1 = nAltValid < %d; 2 = maxAltDca > %.2f cm; 3 = 1 and 2), alternatives evaluated only for q_inv < %.3f",
         crosspair.cfgDoCrossPairCut.value ? "ON" : "off", crosspair.cfgCrossVetoMode.value,
         crosspair.cfgCrossMaxMeeRatio.value, crosspair.cfgAltRequireBoth.value ? 2 : 1,
         crosspair.cfgCrossMinMaxAltDca.value, crosspair.cfgAltMaxQinv.value);
    if (crosspair.cfgDoCrossPairCut.value && crosspair.cfgCrossVetoMode.value >= 2 && crosspair.cfgAltMaxQinv.value < 0.3f) { // o2-linter: disable=magic-number (cross pair on)
      LOGF(warning, "photonhbt: the maxAltDca veto can only fire below cfgAltMaxQinv = %.3f; above it maxAltDca stays at its 'not evaluated' default and every pair passes",
           crosspair.cfgAltMaxQinv.value);
    }
    LOGF(info, "photonhbt: dedup %s", dedup.cfgDoDedup.value ? "ON" : "off");
    mAltCuts.maxDca = crosspair.cfgAltMaxDca.value;
    mAltCuts.minR = crosspair.cfgAltMinR.value;
    mAltCuts.maxR = crosspair.cfgAltMaxR.value;
    mAltCuts.minCosPA = crosspair.cfgAltMinCosPA.value;
    mAltCuts.maxMee = crosspair.cfgAltMaxMee.value;
    mAltCuts.maxDcaXY = crosspair.cfgAltMaxDcaXY.value;
    mAltCuts.maxDcaZ = crosspair.cfgAltMaxDcaZ.value;
    mCandCuts.minPt = pcmcuts.cfgMinPtV0.value;
    mCandCuts.maxEta = pcmcuts.cfgMaxEtaV0.value;
    mCandCuts.minCosPA = pcmcuts.cfgMinCosPA.value;
    mCandCuts.maxPca = pcmcuts.cfgMaxPCA.value;
    mCandCuts.minR = pcmcuts.cfgMinV0Radius.value;
    mCandCuts.maxR = pcmcuts.cfgMaxV0Radius.value;
    mCandCuts.maxAlpha = pcmcuts.cfgMaxAlphaAP.value;
    mCandCuts.maxQt = pcmcuts.cfgMaxQtAP.value;
    if (mixing.cfgMixMode.value < 0 || mixing.cfgMixMode.value > 2) { // o2-linter: disable=magic-number (valid modes)
      LOGF(fatal, "photonhbt: cfgMixMode %d does not exist; 0 = photon-level, 1 = leg-level competition, 2 = leg-built reference", mixing.cfgMixMode.value);
    }
    if (mixing.cfgMixMode.value == 2) { // o2-linter: disable=magic-number (mode 2)
      LOGF(info, "photonhbt: mixed event = leg-built reference: G_AB = e+(A) e-(B), G_BA = e+(B) e-(A), analytic, PCA < %.2f, %.0f < R < %.0f, unique legs by smallest PCA, pairs G_AB x G_BA. No stored photon enters the denominator.",
           mixbuilder.cfgBuilderMaxPca.value, mixbuilder.cfgBuilderMinR.value, mixbuilder.cfgBuilderMaxR.value);
    }
    if (mixing.cfgMixMode.value == 0) {
      LOGF(info, "photonhbt: mixed event = photon-level (built photons paired as they are)");
    } else {
      LOGF(info, "photonhbt: mixed event = leg-level, four-leg candidate competition below q_inv %.3f, score weight %.2f; cross candidates: DCAFitter, builder acceptance (cosPA > %.3f, PCA < %.2f, %.0f < R < %.0f, |alpha| < %.2f, qT < %.3f), analysis cuts on the survivors (pT > %.2f, |eta| < %.2f, cosPA > %.4f, PCA < %.2f, %.1f < R < %.1f, |alpha| < %.2f, qT < %.3f)",
           mixing.cfgQuartetMaxQinv.value, mixing.cfgScoreWeight.value,
           mixbuilder.cfgBuilderMinCosPA.value, mixbuilder.cfgBuilderMaxPca.value, mixbuilder.cfgBuilderMinR.value, mixbuilder.cfgBuilderMaxR.value, mixbuilder.cfgBuilderMaxAlpha.value, mixbuilder.cfgBuilderMaxQt.value,
           mCandCuts.minPt, mCandCuts.maxEta, mCandCuts.minCosPA, mCandCuts.maxPca, mCandCuts.minR, mCandCuts.maxR, mCandCuts.maxAlpha, mCandCuts.maxQt);
    }
    if (mixing.cfgMixMode.value == 1 && mixing.cfgQuartetMaxQinv.value > qaflags.cfgMaxQinvForProcessing.value) {
      LOGF(fatal, "photonhbt: cfgQuartetMaxQinv (%.3f) above cfgMaxQinvForProcessing (%.3f): the processing gate would cut quartets before they are evaluated", mixing.cfgQuartetMaxQinv.value, qaflags.cfgMaxQinvForProcessing.value);
    }
    if (mixing.cfgMixMode.value == 1 && crosspair.cfgDoCrossPairCut.value) {
      LOGF(warning, "photonhbt: the cross-pair veto removes from the mixed event exactly the pairs the four-leg competition adds; switch it off for the correlation function");
    }
  }

  template <typename TCollision>
  void initCCDB(TCollision const& collision)
  {
    if (mRunNumber == collision.runNumber()) {
      return;
    }
    mRunNumber = collision.runNumber();

    auto vd = ccdb->getForRun<o2::tpc::VDriftCorrFact>("TPC/Calib/VDriftTgl", mRunNumber);
    if (vd != nullptr) {
      mVDriftCmPerNs = vd->refVDrift * vd->corrFact * 1e-3f; // o2-linter: disable=magic-number (cm/us -> cm/ns)
      LOGF(info, "photonhbt: run %d, TPC vdrift = %.6f cm/ns (%.4f cm/us); 1 cm in z = %.0f ns",
           mRunNumber, mVDriftCmPerNs, 1e3f * mVDriftCmPerNs,
           (mVDriftCmPerNs > 0.f) ? 1.f / mVDriftCmPerNs : -1.f);
    } else {
      mVDriftCmPerNs = 0.f;
      LOGF(warn, "photonhbt: no TPC VDrift object for run %d -- the z scale stays in cm", mRunNumber);
    }

    if (cfgBzOverrideT.value > -100.f) { // o2-linter: disable=magic-number (number in case B-field is overridden in case not fetched from CCDB)
      mBzT = cfgBzOverrideT.value;
      return;
    }
    auto grpmag = ccdb->getForRun<o2::parameters::GRPMagField>("GLO/Config/GRPMagField", mRunNumber);
    mBzT = 0.1f * static_cast<float>(grpmag->getNominalL3Field());
    LOGF(info, "photonhbt: run %d, Bz = %.2f T", mRunNumber, mBzT);
  }

  void DefineEMEventCut()
  {
    fEMEventCut = EMPhotonEventCut("fEMEventCut", "fEMEventCut");
    fEMEventCut.SetRequireSel8(eventcuts.cfgRequireSel8);
    fEMEventCut.SetRequireFT0AND(eventcuts.cfgRequireFT0AND);
    fEMEventCut.SetZvtxRange(eventcuts.cfgZvtxMin, eventcuts.cfgZvtxMax);
    fEMEventCut.SetRequireNoTFB(eventcuts.cfgRequireNoTFB);
    fEMEventCut.SetRequireNoITSROFB(eventcuts.cfgRequireNoITSROFB);
    fEMEventCut.SetRequireNoSameBunchPileup(eventcuts.cfgRequireNoSameBunchPileup);
    fEMEventCut.SetRequireVertexITSTPC(eventcuts.cfgRequireVertexITSTPC);
    fEMEventCut.SetRequireGoodZvtxFT0vsPV(eventcuts.cfgRequireGoodZvtxFT0vsPV);
    fEMEventCut.SetRequireNoCollInTimeRangeStandard(eventcuts.cfgRequireNoCollInTimeRangeStandard);
    fEMEventCut.SetRequireNoCollInTimeRangeStrict(eventcuts.cfgRequireNoCollInTimeRangeStrict);
    fEMEventCut.SetRequireNoCollInITSROFStandard(eventcuts.cfgRequireNoCollInITSROFStandard);
    fEMEventCut.SetRequireNoCollInITSROFStrict(eventcuts.cfgRequireNoCollInITSROFStrict);
    fEMEventCut.SetRequireNoHighMultCollInPrevRof(eventcuts.cfgRequireNoHighMultCollInPrevRof);
    fEMEventCut.SetRequireGoodITSLayer3(eventcuts.cfgRequireGoodITSLayer3);
    fEMEventCut.SetRequireGoodITSLayer0123(eventcuts.cfgRequireGoodITSLayer0123);
    fEMEventCut.SetRequireGoodITSLayersAll(eventcuts.cfgRequireGoodITSLayersAll);
  }

  void DefinePCMCut()
  {
    fV0PhotonCut = V0PhotonCut("fV0PhotonCut", "fV0PhotonCut");
    fV0PhotonCut.SetV0PtRange(pcmcuts.cfgMinPtV0, 1e10f);
    fV0PhotonCut.SetV0EtaRange(-pcmcuts.cfgMaxEtaV0, +pcmcuts.cfgMaxEtaV0);
    fV0PhotonCut.SetMinCosPA(pcmcuts.cfgMinCosPA);
    fV0PhotonCut.SetMaxPCA(pcmcuts.cfgMaxPCA);
    fV0PhotonCut.SetMaxChi2KF(pcmcuts.cfgMaxChi2KF);
    fV0PhotonCut.SetRxyRange(pcmcuts.cfgMinV0Radius, pcmcuts.cfgMaxV0Radius);
    fV0PhotonCut.SetAPRange(pcmcuts.cfgMaxAlphaAP, pcmcuts.cfgMaxQtAP);
    fV0PhotonCut.RejectITSib(pcmcuts.cfgRejectV0OnITSIB);
    fV0PhotonCut.SetMinNClustersTPC(pcmcuts.cfgMinNClusterTPC);
    fV0PhotonCut.SetMinNCrossedRowsTPC(pcmcuts.cfgMinNCrossedRows);
    fV0PhotonCut.SetMinNCrossedRowsOverFindableClustersTPC(0.8);
    fV0PhotonCut.SetMaxFracSharedClustersTPC(pcmcuts.cfgMaxFracSharedClustersTPC);
    fV0PhotonCut.SetChi2PerClusterTPC(0.0, pcmcuts.cfgMaxChi2TPC);
    fV0PhotonCut.SetTPCNsigmaElRange(pcmcuts.cfgMinTPCNsigmaEl, pcmcuts.cfgMaxTPCNsigmaEl);
    fV0PhotonCut.SetChi2PerClusterITS(-1e+10, pcmcuts.cfgMaxChi2ITS);
    fV0PhotonCut.SetDisableITSonly(pcmcuts.cfgDisableITSOnlyTrack);
    fV0PhotonCut.SetDisableTPConly(pcmcuts.cfgDisableTPCOnlyTrack);
    fV0PhotonCut.SetNClustersITS(0, 7);
    fV0PhotonCut.SetMeanClusterSizeITSob(0.0, 16.0);
    fV0PhotonCut.SetRequireITSTPC(pcmcuts.cfgRequireV0WithITSTPC);
    fV0PhotonCut.SetRequireITSonly(pcmcuts.cfgRequireV0WithITSOnly);
    fV0PhotonCut.SetRequireTPConly(pcmcuts.cfgRequireV0WithTPCOnly);
  }

  /*************************************************/
  // Helpers
  /*************************************************/

  inline bool isInsideEllipse(float deta, float dphi) const
  {
    if (!ggpaircuts.cfgDoEllipseCut.value) {
      return false;
    }
    const float sE = ggpaircuts.cfgEllipseSigEta.value;
    const float sP = ggpaircuts.cfgEllipseSigPhi.value;
    if (sE < kMinSigma || sP < kMinSigma) {
      return false;
    }
    return (deta / sE) * (deta / sE) + (dphi / sP) * (dphi / sP) < ggpaircuts.cfgEllipseR2.value;
  }

  inline bool passRZCut(float deltaR, float deltaZ) const
  {
    if (ggpaircuts.cfgDoRCut.value && deltaR < ggpaircuts.cfgMinDeltaR.value) {
      return false;
    }
    if (ggpaircuts.cfgDoZCut.value && std::fabs(deltaZ) < ggpaircuts.cfgMinDeltaZ.value) {
      return false;
    }
    return true;
  }

  [[nodiscard]] inline bool passPhotonClassPairCut(pairutil::V0PhotonLegCounts const& c1,
                                                   pairutil::V0PhotonLegCounts const& c2) const
  {
    if (!cfgDoPhotonClassPairCut.value) {
      return true;
    }
    return pairutil::isPairPhotonClassSelected(c1, c2, mPhotonClassSelA, mPhotonClassSelB);
  }

  inline bool passAsymmetryCut(float pt1, float pt2) const
  {
    if (ggpaircuts.cfgMaxAsymmetry.value < 0.f) {
      return true;
    }

    const float sum = pt1 + pt2;
    if (sum < kMinSigma) {
      return false;
    }
    return std::fabs(pt1 - pt2) / sum < ggpaircuts.cfgMaxAsymmetry.value;
  }

  /*************************************************/
  // Histograms
  /*************************************************/

  void addhistograms()
  {
    addEventHistograms();
    addPairCFHistograms();
    addCrossPairHistograms();
    if (isMC) {
      addPairMCHistograms();
      addTruthMCHistograms();
    }
  }

  // ─── Event histograms (fRegistry) ─────────────────────────────────────────
  void addEventHistograms()
  {
    static constexpr std::array<std::string_view, 6> det = {"FT0M", "FT0A", "FT0C", "BTot", "BPos", "BNeg"};

    o2::aod::pwgem::photonmeson::utils::eventhistogram::addEventHistograms(&fRegistry);
    fRegistry.add("Event/before/hEP2_CentFT0C_forMix", Form("2nd harmonics EP for mix;centrality FT0C (%%);#Psi_{2}^{%s} (rad.)", det[mixing.cfgEP2EstimatorForMix].data()), kTH2D, {{110, 0, 110}, {180, -o2::constants::math::PIHalf, +o2::constants::math::PIHalf}}, false);
    fRegistry.add("Event/after/hEP2_CentFT0C_forMix", Form("2nd harmonics EP for mix;centrality FT0C (%%);#Psi_{2}^{%s} (rad.)", det[mixing.cfgEP2EstimatorForMix].data()),
                  kTH2D, {{110, 0, 110}, {180, -o2::constants::math::PIHalf, +o2::constants::math::PIHalf}}, false);
  }

  void addPairCFHistograms()
  {
    const AxisSpec axisMultNTracks = makeAxisMultNTracks();
    const AxisSpec axisMultFT0M = makeAxisMultFT0M();
    if (hbtanalysis.cfgDo3D) {
      fRegistryCF.add("Pair/same/CF_3D", "diphoton correlation 3D LCMS", kTHnSparseD, {axisQout, axisQside, axisQlong, axisKt}, true);
      if (hbtanalysis.cfgDo2D) {
        fRegistryCF.add("Pair/same/CF_2D", "diphoton correlation 2D (qout,qinv)", kTHnSparseD, {axisQout, axisQinv, axisMultNTracks, axisMultFT0M, axisKt}, true);
        if (hbtanalysis.cfgDo2DSideLong) {
          fRegistryCF.add("Pair/same/CF_2D_Side", "diphoton correlation 2D (qside,qinv)", kTHnSparseD, {axisQside, axisQinv, axisMultNTracks, axisMultFT0M, axisKt}, true);
          fRegistryCF.add("Pair/same/CF_2D_Long", "diphoton correlation 2D (qlong,qinv)", kTHnSparseD, {axisQlong, axisQinv, axisMultNTracks, axisMultFT0M, axisKt}, true);
        }
      }
      if (hbtanalysis.cfgDoQinvGate3D) {
        const AxisSpec axisQinvGate{{0.0, 0.01, 0.02, 0.03, 0.05, 0.30}, "q_{inv} (GeV/c)"};
        fRegistryCF.add("Pair/same/CF_3D_Qinv", "diphoton correlation 3D LCMS + qinv gate axis",
                        kTHnSparseD, {axisQout, axisQside, axisQlong, axisKt, axisQinvGate}, true);
      }
    } else {
      fRegistryCF.add("Pair/same/CF_1D", hbtanalysis.cfgUseLCMS ? "diphoton correlation 1D LCMS" : "diphoton correlation 1D (qinv)", kTH2D, {hbtanalysis.cfgUseLCMS ? axisQabsLcms : axisQinv, axisKt}, true);
    }
    fRegistryCF.add("Pair/same/hSparse_DEtaDPhi_qinv_kT",
                    "pair (#Delta#eta,#Delta#phi,q_{inv},k_{T}) for efficiency reweighting;"
                    "#Delta#eta;#Delta#phi (rad);q_{inv} (GeV/c);k_{T} (GeV/c)",
                    kTHnSparseF, {axisDeltaEta, axisDeltaPhi, axisQinv, axisKt}, true);
    fRegistryCF.add("Pair/same/CF_QLcms_Qinv", "diphoton CF |q|_{LCMS} vs. q_{inv}", kTHnSparseD, {axisQabsLcms, axisQinv, axisKt}, true);
    fRegistryCF.addClone("Pair/same/", "Pair/mix/");
    fRegistryCF.add("Pair/mix/hDiffBC", "diff. global BC in mixed event;|BC_{current}-BC_{mixed}|", kTH1D, {{10001, -0.5, 10000.5}}, true);
    fRegistryCF.add("Pair/hLegXYZ_dLegs", "V0LegsXYZ: distance of the two legs' stored points;|#vec{x}_{+} - #vec{x}_{-}| (cm);V0s", kTH1D, {{200, 0.f, 100.f}}, true);
    fRegistryCF.add("Pair/hLegXYZ_R_vs_Rconv", "V0LegsXYZ: radius of a leg's point vs the V0 conversion radius;R_{conv} (cm);#sqrt{x^{2}+y^{2}} of the leg point (cm)", kTH2D, {{100, 0.f, 100.f}, {100, 0.f, 100.f}}, true);
    fRegistryCF.add("Pair/hLegXYZ_dLegs_vs_Pca", "V0LegsXYZ: distance of the two legs' points vs the builder's PCA (must be the diagonal);PCA (cm);|#vec{x}_{+} - #vec{x}_{-}| (cm)", kTH2D, {{100, 0.f, 5.f}, {100, 0.f, 5.f}}, true);
    fRegistryCF.add("Pair/hLegXYZ_x_vs_y", "V0LegsXYZ: the stored x and y of a leg (global coordinates);x (cm);y (cm)", kTH2D, {{100, -100.f, 100.f}, {100, -100.f, 100.f}}, true);
    fRegistryCF.add("Pair/hLegXYZ_PcaRebuilt_vs_Pca", "stored V0 rebuilt analytically from its leg state: PCA vs the builder's PCA;PCA (table, cm);PCA (rebuilt, cm)", kTH2D, {{100, 0.f, 5.f}, {100, 0.f, 5.f}}, true);
    fRegistryCF.add("Pair/hLegXYZ_RRebuilt_vs_R", "stored V0 rebuilt analytically from its leg state: R vs the builder's R;R_{conv} (table, cm);R_{conv} (rebuilt, cm)", kTH2D, {{100, 0.f, 100.f}, {100, 0.f, 100.f}}, true);
    fRegistryCF.add("Pair/hLegXYZ_RebuildOk", "stored V0 rebuilt analytically from its leg state: 0 = failed, 1 = ok;rebuild ok", kTH1D, {{2, -0.5f, 1.5f}}, true);
    fRegistryCF.add("Pair/hNPhotonsInPool", "selected photons per event entering the mixing pool;N_{#gamma};events", kTH1D, {{21, -0.5f, 20.5f}}, true);
    if (mixing.cfgMixMode.value == 2) { // o2-linter: disable=magic-number (mode 2)
      fRegistryCF.add("Pair/mix/hLegBuilt_NAB_NBA", "leg-built photons per event pair;N(G_{AB});N(G_{BA})", kTH2D, {{30, -0.5f, 29.5f}, {30, -0.5f, 29.5f}}, true);
      fRegistryCF.add("Pair/mix/hLegBuilt_SameQuartet_vs_Qinv", "leg-built pairs: the two photons come from the same four legs (1) or not (0);q_{inv} (GeV/c);same quartet", kTH2D, {axisQinv, {2, -0.5f, 1.5f}}, true);
      fRegistryCF.add("Pair/mix/hLegBuilt_Pca", "leg-built photons: PCA of the accepted candidates;PCA (cm);photons", kTH1D, {{100, 0.f, 5.f}}, true);
    }
    if (mixing.cfgMixMode.value == 1) {
      const AxisSpec axisEmu{4, -0.5f, 3.5f, "0 = not evaluated, 1 = kept, 2 = swapped, 3 = pair lost"};
      const AxisSpec axQwin{30, 0.f, 0.15f, "q_{inv}(a, b) (GeV/c)"};
      fRegistryCF.add("Pair/mix/hQuartetOutcome", "four-leg candidate competition on mixed pairs;q_{inv}(a, b) (GeV/c);outcome", kTH2D, {axisQinv, axisEmu}, true);
      fRegistryCF.add("Pair/mix/hQuartetQinvCross_vs_Qinv", "swapped mixed pairs: q_{inv} of (x_{1}, x_{2}) vs of (a, b);q_{inv}(a, b) (GeV/c);q_{inv}(x_{1}, x_{2}) (GeV/c)", kTH2D, {axisQinv, axisQinv}, true);
      auto hFail = fRegistryCF.add<TH2>("Pair/mix/hQuartetCrossFail_vs_Qinv", "cross candidates of mixed quartets: which builder cut fails (one entry per failed cut);q_{inv}(a, b) (GeV/c);", kTH2D, {axQwin, {7, -0.5f, 6.5f}}, true);
      for (int i = 0; i < 7; ++i) {                                                                                                                                   // o2-linter: disable=magic-number (seven bins)
        hFail->GetYaxis()->SetBinLabel(i + 1, std::array<const char*, 7>{"passed", "fitter failed", "cosPA", "PCA", "R", "#alpha", "q_{T}"}[static_cast<size_t>(i)]); // o2-linter: disable=magic-number (labels)
      }
      fRegistryCF.add("Pair/mix/hVtxDist_vs_Qinv_Outcome", "mixed quartets: 3D distance of the two conversion points (common frame);q_{inv}(a, b) (GeV/c);outcome;d_{3D} (cm)", kTH3D, {axQwin, axisEmu, {100, 0.f, 50.f}}, true);
      fRegistryCF.add("Pair/mix/hOwnScore_Refit_vs_Table", "AO2D path: score of a stored V0 refitted from its real tracks vs the builder's table values;score (table: pca, cospa);score (refit)", kTH2D, {{200, 0.f, 20.f}, {200, 0.f, 20.f}}, true);
      fRegistryCF.add("Pair/mix/hQuartetDeltaScore_vs_Qinv", "quartets with a cross candidate: min(S_{x1}, S_{x2}) - min(S_{a}, S_{b}), negative = a cross takes the first leg;q_{inv}(a, b) (GeV/c);#DeltaS", kTH2D, {axQwin, {200, -20.f, 20.f}}, true);
      fRegistryCF.add("Pair/mix/hCrossPca_Analytic_vs_Fitter", "cross candidates: PCA analytic vs DCAFitter;PCA fitter (cm);PCA analytic (cm)", kTH2D, {{100, 0.f, 5.f}, {100, 0.f, 5.f}}, true);
      fRegistryCF.add("Pair/mix/hCrossVz_Analytic_minus_Fitter", "cross candidates: z of the vertex, analytic - DCAFitter;PCA fitter (cm);#Deltaz (cm)", kTH2D, {{100, 0.f, 5.f}, {200, -10.f, 10.f}}, true);
    }
  }

  [[nodiscard]] static inline float maxPairDcaZ(PhotonWithLegs const& a, PhotonWithLegs const& b)
  {
    return std::max(std::fabs(a.fDcaZToPV), std::fabs(b.fDcaZToPV));
  }
  [[nodiscard]] static inline float maxPairDcaXY(PhotonWithLegs const& a, PhotonWithLegs const& b)
  {
    return std::max(std::fabs(a.fDcaXYToPV), std::fabs(b.fDcaXYToPV));
  }
  [[nodiscard]] inline bool passPointingPairCut(PhotonWithLegs const& a, PhotonWithLegs const& b) const
  {
    return maxPairDcaZ(a, b) < ggpaircuts.cfgMaxDcaZToPV.value && maxPairDcaXY(a, b) < ggpaircuts.cfgMaxDcaXYToPV.value;
  }

  void addCrossPairHistograms()
  {
    if (!crosspair.cfgDoCrossPairCut.value) {
      return;
    }
    for (const auto& sm : {std::string("Pair/same/CrossPair/"), std::string("Pair/mix/CrossPair/")}) {
      auto h = fRegistryCF.add<TH1>((sm + "hVetoCounter").c_str(), "crossed-pair veto;;pairs", kTH1D, {{2, -0.5f, 1.5f}}, true);
      h->GetXaxis()->SetBinLabel(1, "evaluated");
      h->GetXaxis()->SetBinLabel(2, "vetoed");
    }
  }

  //  CF per truth type
  void addPairMCHistograms()
  {
    const AxisSpec axisMultNTracks = makeAxisMultNTracks();
    const AxisSpec axisMultFT0M = makeAxisMultFT0M();
    const AxisSpec axisTruthType{{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5},
                                 "truth type (1=TrueTrueDistinct,2=TrueTrueSamePhoton,3=SharedMcLeg,"
                                 "4=TrueFake,5=FakeFake,6=Pi0Daughters)"};

    static constexpr std::array<std::string_view, 6> kTypes = {"TrueTrueDistinct/", "TrueTrueSamePhoton/", "SharedMcLeg/", "TrueFake/", "FakeFake/", "Pi0Daughters/"};

    for (const auto& label : kTypes) {
      const std::string base = std::string("Pair/same/MC/") + std::string(label);
      if (hbtanalysis.cfgDo3D) {
        fRegistryPairMC.add((base + "CF_3D").c_str(), "MC CF 3D LCMS", kTHnSparseD, {axisQout, axisQside, axisQlong, axisKt}, true);
        if (hbtanalysis.cfgDo2D) {
          fRegistryPairMC.add((base + "CF_2D").c_str(), "MC CF 2D", kTHnSparseD, {axisQout, axisQinv, axisMultNTracks, axisMultFT0M, axisKt}, true);
        }
      } else {
        fRegistryPairMC.add((base + "CF_1D").c_str(), hbtanalysis.cfgUseLCMS ? "MC CF 1D LCMS" : "MC CF 1D (qinv)", kTH2D, {hbtanalysis.cfgUseLCMS ? axisQabsLcms : axisQinv, axisKt}, true);
      }
      fRegistryPairMC.add((base + "CF_QLcms_Qinv").c_str(), "MC CF |q|_{LCMS} vs. q_{inv}", kTHnSparseD, {axisQabsLcms, axisQinv, axisKt}, true);
    }

    fRegistryPairMC.add("Pair/same/MC/hTruthTypeVsQinv", "truth type vs q_{inv};q_{inv} (GeV/c);truth type", kTH2D, {axisQinv, axisTruthType}, true);
    fRegistryPairMC.add("Pair/same/MC/hTruthTypeVsKt", "truth type vs k_{T};k_{T} (GeV/c);truth type", kTH2D, {axisKt, axisTruthType}, true);
    {
      const AxisSpec axisEmu{4, -0.5f, 3.5f, "0 = not evaluated, 1 = kept, 2 = swapped, 3 = pair lost"};
      fRegistryPairMC.add("Pair/same/MC/hQuartetOutcome_vs_Truth", "closure: the quartet competition replayed on stored same-event pairs (a full-information emulator gives kept for all);q_{inv} (GeV/c);truth type;outcome", kTH3D, {axisQinv, axisTruthType, axisEmu}, true);
      auto hFailSE = fRegistryPairMC.add<TH2>("Pair/same/MC/hQuartetCrossFail_vs_Qinv", "closure: cross candidates of stored same-event pairs: which builder cut fails (one entry per failed cut);q_{inv} (GeV/c);", kTH2D, {{30, 0.f, 0.15f}, {7, -0.5f, 6.5f}}, true);
      for (int i = 0; i < 7; ++i) {                                                                                                                                     // o2-linter: disable=magic-number (seven bins)
        hFailSE->GetYaxis()->SetBinLabel(i + 1, std::array<const char*, 7>{"passed", "fitter failed", "cosPA", "PCA", "R", "#alpha", "q_{T}"}[static_cast<size_t>(i)]); // o2-linter: disable=magic-number (labels)
      }
      fRegistryPairMC.add("Pair/same/MC/hTransfer_vs_Qorig", "builder transfer in the same event at the original q;q_{inv} of the true photon pair (GeV/c);0 = kept (TrueTrueDistinct), 1 = swapped (full-cross FakeFake)", kTH2D, {{30, 0.f, 0.15f}, {2, -0.5f, 1.5f}}, true);
      fRegistryPairMC.add("Pair/same/MC/hVtxDist_vs_Qinv_Type", "same-event pairs: 3D distance of the two conversion points;q_{inv} (GeV/c);truth type;d_{3D} (cm)", kTH3D, {{30, 0.f, 0.15f}, axisTruthType, {100, 0.f, 50.f}}, true);
      fRegistryPairMC.add("Pair/same/MC/hQuartetDeltaScore_vs_Truth", "closure: min(S_{x1}, S_{x2}) - min(S_{a}, S_{b}) on stored same-event pairs;q_{inv} (GeV/c);truth type;#DeltaS", kTH3D, {axisQinv, axisTruthType, {200, -20.f, 20.f}}, true);
    }

    for (const auto& label : {std::string("TrueTrueDistinct/"), std::string("TrueFake/"), std::string("FakeFake/")}) {
      const std::string mixBase = "Pair/mix/MC/" + label;
      if (hbtanalysis.cfgDo3D) {
        fRegistryPairMC.add((mixBase + "CF_3D").c_str(), "MC mixed CF 3D LCMS", kTHnSparseD, {axisQout, axisQside, axisQlong, axisKt}, true);
        if (hbtanalysis.cfgDo2D) {
          fRegistryPairMC.add((mixBase + "CF_2D").c_str(), "MC mixed CF 2D", kTHnSparseD, {axisQout, axisQinv, axisMultNTracks, axisMultFT0M, axisKt}, true);
        }
      } else {
        fRegistryPairMC.add((mixBase + "CF_1D").c_str(), hbtanalysis.cfgUseLCMS ? "MC mixed CF 1D LCMS" : "MC mixed CF 1D (qinv)", kTH2D, {hbtanalysis.cfgUseLCMS ? axisQabsLcms : axisQinv, axisKt}, true);
      }
      fRegistryPairMC.add((mixBase + "CF_QLcms_Qinv").c_str(), "MC mixed CF |q|_{LCMS} vs. q_{inv}", kTHnSparseD, {axisQabsLcms, axisQinv, axisKt}, true);
    }

    if (hbtanalysis.cfgDo3D) {
      fRegistryPairMC.add("Pair/same/MC/NoLabel/CF_3D", "missing MC label - CF 3D LCMS", kTHnSparseD, {axisQout, axisQside, axisQlong, axisKt}, true);
      if (hbtanalysis.cfgDo2D) {
        fRegistryPairMC.add("Pair/same/MC/NoLabel/CF_2D", "missing MC label - CF 2D", kTHnSparseD, {axisQout, axisQinv, axisMultNTracks, axisMultFT0M, axisKt}, true);
      }
    } else {
      fRegistryPairMC.add("Pair/same/MC/NoLabel/CF_1D", hbtanalysis.cfgUseLCMS ? "missing MC label - CF 1D LCMS" : "missing MC label - CF 1D (qinv)", kTH2D, {hbtanalysis.cfgUseLCMS ? axisQabsLcms : axisQinv, axisKt}, true);
    }
  }

  void addTruthMCHistograms()
  {
    const AxisSpec axQinvMC{60, 0.f, 0.3f, "q_{inv}^{true} (GeV/c)"};

    if (mctruth.cfgDoTruthMix.value) {
      fRegistryTruthMC.add("MC/TruthCF/hQinvVsKt_same", "truth-level same-event CF;k_{T} (GeV/c);q_{inv}^{true} (GeV/c)", kTH2D, {axisKt, axQinvMC}, true);
      fRegistryTruthMC.add("MC/TruthCF/hDEtaDPhi_same",
                           "truth-level same-event #Delta#eta vs #Delta#phi;"
                           "#Delta#eta_{#gamma#gamma};#Delta#phi_{#gamma#gamma} (rad)",
                           kTH2D, {axisDeltaEta, axisDeltaPhi}, true);
      fRegistryTruthMC.add("MC/TruthCF/hQinvVsKt_mix", "truth-level mixed-event CF;k_{T} (GeV/c);q_{inv}^{true} (GeV/c)", kTH2D, {axisKt, axQinvMC}, true);
      fRegistryTruthMC.add("MC/TruthCF/hDEtaDPhi_mix",
                           "truth-level mixed-event #Delta#eta vs #Delta#phi;"
                           "#Delta#eta_{#gamma#gamma};#Delta#phi_{#gamma#gamma} (rad)",
                           kTH2D, {axisDeltaEta, axisDeltaPhi}, true);
      if (mctruth.cfgDoTruthLcms.value) {
        for (const auto& sm : {std::string("same"), std::string("mix")}) {
          fRegistryTruthMC.add(("MC/TruthCF/hSparse_Qout_Qinv_Kt_" + sm).c_str(), "truth-level CF, LCMS out-component;q_{out}^{true} (GeV/c);q_{inv}^{true} (GeV/c);k_{T} (GeV/c)", kTHnSparseD, {axisQout, axisQinv, axisKt}, true);
        }
      }
      if (mctruth.cfgDoTruth3D.value) {
        for (const auto& sm : {std::string("same"), std::string("mix")}) {
          fRegistryTruthMC.add(("MC/TruthCF/hSparse_Qout_Qside_Qlong_Kt_" + sm).c_str(), "truth-level 3D LCMS CF;q_{out}^{true} (GeV/c);q_{side}^{true} (GeV/c);q_{long}^{true} (GeV/c);k_{T} (GeV/c)", kTHnSparseD, {axisQout, axisQside, axisQlong, axisKt}, true);
        }
      }
    }

    if (mctruth.cfgDoPairEff.value) {
      static const std::array<const char*, 4> kEffStage = {"converted", "legsInV0", "v0Built", "v0Selected"};
      for (const auto& sm : {std::string("same"), std::string("mix")}) {
        for (const auto& st : kEffStage) {
          fRegistryTruthMC.add(("MC/PairEff/" + sm + "/hSparse_Qout_Qinv_Kt_" + st).c_str(), Form("pair efficiency stage %s [%s], at true kinematics;q_{out}^{true} (GeV/c);q_{inv}^{true} (GeV/c);k_{T} (GeV/c)", st, sm.c_str()), kTHnSparseD, {axisQout, axisQinv, axisKt}, true);
        }
      }
      for (const auto& st : kEffStage) {
        fRegistryTruthMC.add(("MC/PairEff/same/hSparse_Qtrue_Rmean_Dconv_" + std::string(st)).c_str(), Form("pair efficiency stage %s vs conversion location;q_{inv}^{true} (GeV/c);(R_{1}^{true} + R_{2}^{true})/2 (cm);|#vec{x}_{1} - #vec{x}_{2}|^{true} (cm)", st), kTHnSparseD, {{20, 0.f, 0.1f}, {50, 0.f, 100.f}, {40, 0.f, 20.f}}, true);
      }
    }
  }

  static inline float computeCosTheta(const ROOT::Math::PtEtaPhiMVector& v1,
                                      const ROOT::Math::PtEtaPhiMVector& v2)
  {
    ROOT::Math::PxPyPzEVector p1(v1), p2(v2);
    ROOT::Math::PxPyPzEVector pair = p1 + p2;
    ROOT::Math::Boost boost(-pair.BoostToCM());
    ROOT::Math::PxPyPzEVector p1cm = boost(p1);
    ROOT::Math::XYZVector pairDir(pair.Px(), pair.Py(), pair.Pz());
    ROOT::Math::XYZVector p1cmDir(p1cm.Px(), p1cm.Py(), p1cm.Pz());
    if (pairDir.R() < kMinSigma || p1cmDir.R() < kMinSigma) {
      return -1.f;
    }
    return static_cast<float>(pairDir.Unit().Dot(p1cmDir.Unit()));
  }

  static void parseBins(const ConfigurableAxis& cfg, std::vector<float>& edges)
  {
    if (cfg.value[0] == VARIABLE_WIDTH) {
      edges = std::vector<float>(cfg.value.begin(), cfg.value.end());
      edges.erase(edges.begin());
    } else {
      const auto n = static_cast<int>(cfg.value[0]);
      const auto xmin = static_cast<float>(cfg.value[1]);
      const auto xmax = static_cast<float>(cfg.value[2]);
      edges.resize(n + 1);
      for (int i = 0; i <= n; ++i) {
        edges[i] = xmin + (xmax - xmin) / n * i;
      }
    }
  }

  static int clampBin(int b, int nmax) { return std::clamp(b, 0, nmax); }

  static int binOf(const std::vector<float>& edges, float val)
  {
    const auto pos = static_cast<int>(std::lower_bound(edges.begin(), edges.end(), val) - edges.begin());
    return clampBin(pos - 1, static_cast<int>(edges.size()) - 2);
  }

  template <PairTruthType TruthT, bool IsMix = false>
  static constexpr const char* mcDirPrefix()
  {
    if constexpr (!IsMix) {
      if constexpr (TruthT == PairTruthType::TrueTrueDistinct) {
        return "Pair/same/MC/TrueTrueDistinct/";
      } else if constexpr (TruthT == PairTruthType::TrueTrueSamePhoton) {
        return "Pair/same/MC/TrueTrueSamePhoton/";
      } else if constexpr (TruthT == PairTruthType::SharedMcLeg) {
        return "Pair/same/MC/SharedMcLeg/";
      } else if constexpr (TruthT == PairTruthType::TrueFake) {
        return "Pair/same/MC/TrueFake/";
      } else if constexpr (TruthT == PairTruthType::FakeFake) {
        return "Pair/same/MC/FakeFake/";
      } else {
        return "Pair/same/MC/Pi0Daughters/";
      }
    } else {
      if constexpr (TruthT == PairTruthType::TrueTrueDistinct) {
        return "Pair/mix/MC/TrueTrueDistinct/";

      } else if constexpr (TruthT == PairTruthType::TrueTrueSamePhoton) {
        return "Pair/mix/MC/TrueTrueSamePhoton/";
      } else if constexpr (TruthT == PairTruthType::SharedMcLeg) {
        return "Pair/mix/MC/SharedMcLeg/";
      } else if constexpr (TruthT == PairTruthType::TrueFake) {
        return "Pair/mix/MC/TrueFake/";

      } else if constexpr (TruthT == PairTruthType::FakeFake) {
        return "Pair/mix/MC/FakeFake/";
      } else {
        return "Pair/mix/MC/Pi0Daughters/";
      }
    }
  }

  template <typename TGamma, typename TLeg>
  [[nodiscard]] PhotonWithLegs makePhotonWithLegs(TGamma const& g, TLeg const& pos, TLeg const& ele, float vtxX, float vtxY, float vtxZ) const
  {
    PhotonWithLegs p;
    p.fVtxX = vtxX;
    p.fVtxY = vtxY;
    p.fVtxZ = vtxZ;
    p.fPt = g.pt();
    p.fEta = g.eta();
    p.fPhi = g.phi();
    p.fVx = g.vx();
    p.fVy = g.vy();
    p.fVz = g.vz();
    p.fLegPt = {static_cast<float>(pos.pt()), static_cast<float>(ele.pt())};
    p.fLegEta = {static_cast<float>(pos.eta()), static_cast<float>(ele.eta())};
    p.fLegPhi = {static_cast<float>(pos.phi()), static_cast<float>(ele.phi())};
    p.fLegNsigEl = {static_cast<float>(pos.tpcNSigmaEl()), static_cast<float>(ele.tpcNSigmaEl())};
    p.fLegNClsFindable = {static_cast<float>(pos.tpcNClsFindable()), static_cast<float>(ele.tpcNClsFindable())};
    p.fLegNClsITS = {static_cast<float>(pos.itsNCls()), static_cast<float>(ele.itsNCls())};
    p.fDcaXYToPV = static_cast<float>(g.dcaXYtopv());
    p.fDcaZToPV = static_cast<float>(g.dcaZtopv());
    p.fLegDcaZ = {static_cast<float>(pos.dcaZ()), static_cast<float>(ele.dcaZ())};
    p.fLegCounts = pairutil::getV0PhotonLegCounts(pos, ele);
    p.fGlobalIndex = g.globalIndex();
    p.fLegTrackId = {pos.trackId(), ele.trackId()};
    auto detMask = [](auto const& l) -> uint8_t {
      return static_cast<uint8_t>((l.hasITS() ? 1 : 0) | (l.hasTPC() ? 2 : 0) | (l.hasTRD() ? 4 : 0) | (l.hasTOF() ? 8 : 0)); // o2-linter: disable=magic-number (bit mask)
    };
    p.fLegDetMask = {detMask(pos), detMask(ele)};
    p.fLegZ0 = {p.fVz, p.fVz};
    if constexpr (requires { pos.x(); pos.y(); pos.z(); }) {
      p.fLegXYZ = {{{static_cast<float>(pos.x()), static_cast<float>(pos.y()), static_cast<float>(pos.z())}, {static_cast<float>(ele.x()), static_cast<float>(ele.y()), static_cast<float>(ele.z())}}};
      p.fHasLegXYZ = true;
    }
    p.fNITSTPC = p.fLegCounts.nITSTPC;
    p.fNForeignLegs = ((pos.collisionId() != g.collisionId()) ? 1 : 0) + ((ele.collisionId() != g.collisionId()) ? 1 : 0);
    p.fNTimedLegs = ((pos.hasITS() || pos.hasTRD() || pos.hasTOF()) ? 1 : 0) + ((ele.hasITS() || ele.hasTRD() || ele.hasTOF()) ? 1 : 0);
    for (size_t ir = 0; ir < kPhiStarRadiiM.size(); ++ir) {
      for (int il = 0; il < 2; ++il) { // o2-linter: disable=magic-number (two legs)
        const auto h = legHelixAt(p.fVx / kCmPerM, p.fVy / kCmPerM, p.fLegPhi[il], p.fLegPt[il], (il == 0) ? +1 : -1, mBzT, kPhiStarRadiiM[ir]);
        p.fLegPhiStar[il][ir] = h.phiStar;
        p.fLegSxy[il][ir] = h.sxy;
      }
    }
    p.fAnaScore = analyticScoreV0(std::clamp(static_cast<float>(g.cospa()), -1.f, 1.f), static_cast<float>(g.pca()), mixing.cfgScoreWeight.value);
    p.fAnaPca = static_cast<float>(g.pca());
    p.fAnaCosPA = std::clamp(static_cast<float>(g.cospa()), -1.f, 1.f);
    p.fAnaDcaXY = std::fabs(p.fDcaXYToPV);
    p.fAnaDcaZ = std::fabs(p.fDcaZToPV);
    p.fAnaPhotonLike = true;
    return p;
  }

  template <typename THist>
  static void fillFailMask(THist& registry, const auto& hist, int mask, float qinv)
  {
    if (mask == 0) {
      registry.fill(hist, qinv, 0.f);
      return;
    }
    for (int bit = 0; bit < 6; ++bit) { // o2-linter: disable=magic-number (six cuts)
      if (mask & (1 << bit)) {
        registry.fill(hist, qinv, static_cast<float>(bit + 1));
      }
    }
  }

  [[nodiscard]] static LegParam legOfAtPoint(PhotonWithLegs const& p, int il)
  {
    LegParam l;
    const auto& q = p.fLegXYZ[static_cast<size_t>(il)];
    l.xyz = {q[0] - p.fVtxX, q[1] - p.fVtxY, q[2] - p.fVtxZ};
    const float pt = p.fLegPt[il];
    l.mom = {pt * std::cos(p.fLegPhi[il]), pt * std::sin(p.fLegPhi[il]), pt * std::sinh(p.fLegEta[il])};
    l.charge = (il == 0) ? +1 : -1;
    return l;
  }

  [[nodiscard]] static LegParam legOf(PhotonWithLegs const& p, int il)
  {
    LegParam l;
    l.xyz = {p.fVx - p.fVtxX, p.fVy - p.fVtxY, p.fVz - p.fVtxZ};
    const float pt = p.fLegPt[il];
    l.mom = {pt * std::cos(p.fLegPhi[il]), pt * std::sin(p.fLegPhi[il]), pt * std::sinh(p.fLegEta[il])};
    l.charge = (il == 0) ? +1 : -1;
    return l;
  }

  [[nodiscard]] Emulation emulateCrossPairing(PhotonWithLegs const& a, PhotonWithLegs const& b, float qinvAB)
  {
    Emulation e;
    if (qinvAB >= mixing.cfgQuartetMaxQinv.value || a.fAnaScore < 0.f || b.fAnaScore < 0.f) {
      return e; // outcome 0
    }
    const float w = mixing.cfgScoreWeight.value;
    const LegParam aPos = legOf(a, 0), aNeg = legOf(a, 1), bPos = legOf(b, 0), bNeg = legOf(b, 1);
    float sA = a.fAnaScore, sB = b.fAnaScore;
    if (a.fHasTrackCov && b.fHasTrackCov) {
      std::array<o2::track::TrackParCov, 2> aTrk = a.fLegTrack;
      std::array<o2::track::TrackParCov, 2> bTrk = b.fLegTrack;
      for (int il = 0; il < 2; ++il) { // o2-linter: disable=magic-number (two legs)
        translateTrackParCov(aTrk[static_cast<size_t>(il)], -a.fVtxX, -a.fVtxY, -a.fVtxZ);
        translateTrackParCov(bTrk[static_cast<size_t>(il)], -b.fVtxX, -b.fVtxY, -b.fVtxZ);
      }
      const AnalyticV0 ownA = buildFitterV0(aTrk[0], aTrk[1], mBzT, w);
      const AnalyticV0 ownB = buildFitterV0(bTrk[0], bTrk[1], mBzT, w);
      if (!ownA.ok || !ownB.ok) {
        return e;
      }
      sA = ownA.score;
      sB = ownB.score;
      e.x1 = buildFitterV0(aTrk[0], bTrk[1], mBzT, w);
      e.x2 = buildFitterV0(bTrk[0], aTrk[1], mBzT, w);
      if (mixing.cfgMixMode.value == 1) {
        fRegistryCF.fill(HIST("Pair/mix/hOwnScore_Refit_vs_Table"), std::min(a.fAnaScore, 19.99f), std::min(ownA.score, 19.99f)); // o2-linter: disable=magic-number (clamp to the axis)
        fRegistryCF.fill(HIST("Pair/mix/hOwnScore_Refit_vs_Table"), std::min(b.fAnaScore, 19.99f), std::min(ownB.score, 19.99f)); // o2-linter: disable=magic-number (clamp to the axis)
      }
    } else if (a.fHasLegXYZ && b.fHasLegXYZ) {
      const LegParam aP = legOfAtPoint(a, 0), aN = legOfAtPoint(a, 1), bP = legOfAtPoint(b, 0), bN = legOfAtPoint(b, 1);
      auto build = [&](LegParam const& pos, LegParam const& neg) {
        return (mixing.cfgCrossBuilder.value == 0) ? buildAnalyticV0(pos, neg, 10.f * mBzT, w) : buildFitterV0(pos, neg, mBzT, w); // o2-linter: disable=magic-number (Tesla -> kGauss)
      };
      const AnalyticV0 ownA = build(aP, aN);
      const AnalyticV0 ownB = build(bP, bN);
      if (!ownA.ok || !ownB.ok) {
        return e;
      }
      sA = ownA.score;
      sB = ownB.score;
      e.x1 = build(aP, bN);
      e.x2 = build(bP, aN);
      if (mixing.cfgMixMode.value == 1) {
        fRegistryCF.fill(HIST("Pair/mix/hOwnScore_Refit_vs_Table"), std::min(a.fAnaScore, 19.99f), std::min(ownA.score, 19.99f)); // o2-linter: disable=magic-number (clamp to the axis)
        fRegistryCF.fill(HIST("Pair/mix/hOwnScore_Refit_vs_Table"), std::min(b.fAnaScore, 19.99f), std::min(ownB.score, 19.99f)); // o2-linter: disable=magic-number (clamp to the axis)
      }
    } else if (mixing.cfgCrossBuilder.value == 0) {
      e.x1 = buildAnalyticV0(aPos, bNeg, 10.f * mBzT, w); // o2-linter: disable=magic-number (Tesla -> kGauss)
      e.x2 = buildAnalyticV0(bPos, aNeg, 10.f * mBzT, w); // o2-linter: disable=magic-number (Tesla -> kGauss)
    } else {
      e.x1 = buildFitterV0(aPos, bNeg, mBzT, w);
      e.x2 = buildFitterV0(bPos, aNeg, mBzT, w);
    }
    const bool fitterRan = (a.fHasTrackCov && b.fHasTrackCov) || mixing.cfgCrossBuilder.value == 1;
    for (const auto& x : {&e.x1, &e.x2}) {
      if (!x->ok || !fitterRan) {
        continue;
      }
      const AnalyticV0 an = (x == &e.x1) ? buildAnalyticV0(aPos, bNeg, 10.f * mBzT, w) : buildAnalyticV0(bPos, aNeg, 10.f * mBzT, w); // o2-linter: disable=magic-number (Tesla -> kGauss)
      if (an.ok && mixing.cfgMixMode.value == 1) {
        fRegistryCF.fill(HIST("Pair/mix/hCrossPca_Analytic_vs_Fitter"), std::min(x->pca, 4.99f), std::min(an.pca, 4.99f));                            // o2-linter: disable=magic-number (clamp to the axis)
        fRegistryCF.fill(HIST("Pair/mix/hCrossVz_Analytic_minus_Fitter"), std::min(x->pca, 4.99f), std::clamp(an.vtx[2] - x->vtx[2], -9.99f, 9.99f)); // o2-linter: disable=magic-number (clamp to the axis)
      }
    }
    e.failMask = {builderFailMask(e.x1), builderFailMask(e.x2)};
    const bool ok1 = (e.failMask[0] == 0);
    const bool ok2 = (e.failMask[1] == 0);
    e.sStoredBest = std::min(sA, sB);
    e.sCrossBest = std::min(ok1 ? e.x1.score : 999.f, ok2 ? e.x2.score : 999.f); // o2-linter: disable=magic-number (absent candidate)
    e.crossExists = ok1 || ok2;
    if (!e.crossExists) {
      e.outcome = 1;
      return e;
    }
    e.dSBest = e.sCrossBest - e.sStoredBest;
    constexpr int kA = 0, kB = 1, kX1 = 2, kX2 = 3;
    const std::array<float, 4> score{sA, sB, ok1 ? e.x1.score : 999.f, ok2 ? e.x2.score : 999.f}; // o2-linter: disable=magic-number (absent candidate)
    const std::array<bool, 4> present{true, true, ok1, ok2};
    std::array<int, 4> order{kA, kB, kX1, kX2};
    std::sort(order.begin(), order.end(), [&score](int i, int j) {
      const float si = score[static_cast<size_t>(i)], sj = score[static_cast<size_t>(j)];
      return (si != sj) ? (si < sj) : (i < j);
    });
    auto shareLeg = [](int i, int j) {
      const bool aa = (i == kA && (j == kX1 || j == kX2)) || (j == kA && (i == kX1 || i == kX2));
      const bool bb = (i == kB && (j == kX1 || j == kX2)) || (j == kB && (i == kX1 || i == kX2));
      return aa || bb;
    };
    std::array<bool, 4> alive{false, false, false, false};
    for (const int& i : order) {
      if (!present[static_cast<size_t>(i)]) {
        continue;
      }
      bool free = true;
      for (int k = 0; k < 4; ++k) { // o2-linter: disable=magic-number (four candidates)
        if (alive[static_cast<size_t>(k)] && shareLeg(i, k)) {
          free = false;
          break;
        }
      }
      alive[static_cast<size_t>(i)] = free;
    }
    if (alive[kA] && alive[kB]) {
      e.outcome = 1;
    } else if (alive[kX1] && alive[kX2] && passCandidateCuts(e.x1) && passCandidateCuts(e.x2)) {
      e.outcome = 2; // o2-linter: disable=magic-number (outcome code, see Emulation)
      e.swapped = true;
    } else {
      e.outcome = 3; // o2-linter: disable=magic-number (outcome code, see Emulation)
    }
    return e;
  }

  [[nodiscard]] static pairutil::V0PhotonLegCounts legCountsFromMasks(uint8_t m0, uint8_t m1)
  {
    pairutil::V0PhotonLegCounts c;
    for (const uint8_t& m : {m0, m1}) {
      const bool its = (m & 1) != 0, tpc = (m & 2) != 0; // o2-linter: disable=magic-number (bit mask)
      if (its && tpc) {
        c.nITSTPC++;
      } else if (its) {
        c.nITSOnly++;
      } else {
        c.nTPCOnly++;
      }
      c.nTRD += ((m & 4) != 0) ? 1 : 0; // o2-linter: disable=magic-number (bit mask)
      c.nTOF += ((m & 8) != 0) ? 1 : 0; // o2-linter: disable=magic-number (bit mask)
    }
    return c;
  }

  [[nodiscard]] static CrossPhotonLite crossPhotonLite(AnalyticV0 const& x, PhotonWithLegs const& frameOf)
  {
    CrossPhotonLite c;
    c.fPt = x.pt();
    c.fEta = x.eta();
    c.fPhi = x.phi();
    c.fVx = x.vtx[0] + frameOf.fVtxX;
    c.fVy = x.vtx[1] + frameOf.fVtxY;
    c.fVz = x.vtx[2] + frameOf.fVtxZ;
    c.fDcaXYToPV = x.dcaxy;
    c.fDcaZToPV = x.dcaz;
    return c;
  }

  [[nodiscard]] PairSep computePairSep(PhotonWithLegs const& a, PhotonWithLegs const& b) const
  {
    PairSep s;
    s.dVtxZ = a.fVtxZ - b.fVtxZ;
    s.nLegsITSTPC = a.fNITSTPC + b.fNITSTPC;

    const float sigR = std::max(pairsep.cfgSigRPhi.value, kMinSigma);
    const float sigZcm = (pairsep.cfgSigZNs.value > 0.f && mVDriftCmPerNs > 0.f)
                           ? pairsep.cfgSigZNs.value * mVDriftCmPerNs
                           : pairsep.cfgSigZ.value;
    const float sigZ = std::max(sigZcm, kMinSigma);
    const float dClose = pairsep.cfgDPairClose.value;
    for (int i = 0; i < 2; ++i) { // o2-linter: disable=magic-number (PP and NN)
      int nHere = 0, nSame = 0, nCloseD = 0;
      std::array<int, 4> nCloseThr{};
      float best3 = 999.f, bestR = 999.f, bestD = 999.f;
      float bestRAll = 999.f, bestDAll = 999.f;
      float rphiAt3 = 999.f, zAt3 = 999.f, zAtR = 999.f;
      float rphiAtD = 999.f, zAtD = 999.f, rAtD = -1.f, absZAtD = -1.f;
      float zGlobAtD = 999.f, zSgnLocAtD = 999.f, zSgnGlobAtD = 999.f;
      float driftAtD = -1.f, driftOldAtD = -1.f;
      for (size_t ir = 0; ir < kPhiStarRadiiM.size(); ++ir) {
        const float p1 = a.fLegPhiStar[i][ir], p2 = b.fLegPhiStar[i][ir];
        if (p1 < -100.f || p2 < -100.f) { // o2-linter: disable=magic-number (invalid phi*)
          continue;
        }
        const float rCm = kPhiStarRadiiM[ir] * kCmPerM;
        const float dphi = RecoDecay::constrainAngle(p1 - p2, -o2::constants::math::PI);
        const float dRPhi = 2.f * rCm * std::fabs(std::sin(0.5f * dphi));
        const float z1 = a.fLegZ0[i] + a.fLegSxy[i][ir] * kCmPerM * std::sinh(a.fLegEta[i]);
        const float z2 = b.fLegZ0[i] + b.fLegSxy[i][ir] * kCmPerM * std::sinh(b.fLegEta[i]);
        const float dZSignedLocal = (z1 - a.fVtxZ) - (z2 - b.fVtxZ);
        const float dZSignedGlobal = z1 - z2;
        const float dZ = std::fabs(dZSignedLocal);
        const float dZGlobal = std::fabs(dZSignedGlobal);
        const float d3 = std::hypot(dRPhi, dZ);
        const float zMax = pairsep.cfgTpcHalfLengthCm.value;
        const float l1 = std::max(0.f, zMax - std::fabs(z1));
        const float l2 = std::max(0.f, zMax - std::fabs(z2));
        const float driftHere = 0.5f * (l1 + l2);
        const float driftOldHere = std::max(0.f, zMax - 0.5f * std::fabs(z1 + z2));
        const bool sameSide = (z1 * z2 >= 0.f);
        const float dPair = std::hypot(dRPhi / sigR, dZ / sigZ);
        ++nHere;
        s.ptR[i][nHere - 1] = rCm;
        s.ptDRPhi[i][nHere - 1] = dRPhi;
        s.ptDZ[i][nHere - 1] = dZ;
        s.ptDZSgnLocal[i][nHere - 1] = dZSignedLocal;
        s.ptDrift[i][nHere - 1] = driftHere;
        s.ptSameSide[i][nHere - 1] = sameSide ? 1.f : 0.f;
        bestRAll = std::min(bestRAll, dRPhi);
        bestDAll = std::min(bestDAll, dPair);
        if (d3 < best3) {
          best3 = d3;
          rphiAt3 = dRPhi;
          zAt3 = dZ;
        }
        if (!sameSide) {
          continue;
        }
        ++nSame;
        for (std::size_t t = 0; t < s.nCloseRPhiThr.size(); ++t) {
          if (dRPhi < pairsep.cfgCloseThrCm.value[t]) {
            ++nCloseThr[t];
          }
        }
        if (dPair < dClose) {
          ++nCloseD;
        }
        if (dPair < bestD) {
          bestD = dPair;
          rphiAtD = dRPhi;
          zAtD = dZ;
          zGlobAtD = dZGlobal;
          zSgnLocAtD = dZSignedLocal;
          zSgnGlobAtD = dZSignedGlobal;
          rAtD = rCm;
          absZAtD = 0.5f * std::fabs(z1 + z2);
          driftAtD = driftHere;
          driftOldAtD = driftOldHere;
        }
        if (dRPhi < bestR) {
          bestR = dRPhi;
          zAtR = dZ;
        }
      }
      if (nHere == 0) {
        continue;
      }
      s.nPoints[i] = nHere;
      s.nSameSidePerCharge[i] = nSame;
      s.dMin3DPerCharge[i] = best3;
      s.dPairMinPerCharge[i] = bestD;
      s.dMinRPhiPerCharge[i] = bestR;
      s.dMinRPhiAll = std::min(s.dMinRPhiAll, bestRAll);
      s.dPairMinAll = std::min(s.dPairMinAll, bestDAll);
      const float invS = (nSame > 0) ? 1.f / static_cast<float>(nSame) : 0.f;
      for (std::size_t t = 0; t < nCloseThr.size(); ++t) {
        s.nCloseRPhiThrPerCharge[i][t] = nCloseThr[t];
        s.fCloseRPhiThrPerCharge[i][t] = static_cast<float>(nCloseThr[t]) * invS;
      }
      s.fCloseRPhiPerCharge[i] = s.fCloseRPhiThrPerCharge[i][3]; // o2-linter: disable=magic-number (last threshold)
      if (bestD < s.dPairMin) {
        s.dPairMin = bestD;
        s.dRPhiAtDPairMin = rphiAtD;
        s.dZAtDPairMin = zAtD;
        s.dZGlobalAtDPairMin = zGlobAtD;
        s.dZSignedLocalAtDPairMin = zSgnLocAtD;
        s.dZSignedGlobalAtDPairMin = zSgnGlobAtD;
        s.rAtDPairMin = rAtD;
        s.absZAtDPairMin = absZAtD;
        s.driftLen = driftAtD;
        s.driftOldAtDPairMin = driftOldAtD;
        s.nCloseScaled = nCloseD;
        s.fCloseScaled = static_cast<float>(nCloseD) * invS;
      }
      if (best3 < s.dMin3D) {
        s.dMin3D = best3;
        s.dRPhiAtMin3D = rphiAt3;
        s.dZAtMin3D = zAt3;
        s.criticalCharge = i;
      }
      if (bestR < s.dMinRPhi) {
        s.dMinRPhi = bestR;
        s.dZAtMinRPhi = zAtR;
        s.nCloseRPhiThr = s.nCloseRPhiThrPerCharge[i];
        s.fCloseRPhiThr = s.fCloseRPhiThrPerCharge[i];
        s.fCloseRPhi = s.fCloseRPhiThr[3]; // o2-linter: disable=magic-number (last threshold)
        s.nSameSide = nSame;
        s.rphiCharge = i;
      }
    }
    s.sameSideAtDPairMin = (s.nSameSidePerCharge[0] + s.nSameSidePerCharge[1]) > 0;
    s.nCommon = (s.rphiCharge >= 0) ? s.nPoints[s.rphiCharge]
                                    : std::max(s.nPoints[0], s.nPoints[1]);
    return s;
  }

  template <typename TG1, typename TG2>
  [[nodiscard]] inline bool passFastQinvGate(TG1 const& g1, TG2 const& g2) const
  {
    const float qMax = qaflags.cfgMaxQinvForProcessing.value;
    if (qMax > 1e9f) { // o2-linter: disable=magic-number (skip if non-sensical value is chosen)
      return true;
    }
    const float dEta = g1.eta() - g2.eta();
    const float dPhi = RecoDecay::constrainAngle(g1.phi() - g2.phi(), -o2::constants::math::PI);
    const float q2 = 2.f * g1.pt() * g2.pt() * (std::cosh(dEta) - std::cos(dPhi));
    return q2 <= qMax * qMax;
  }

  [[nodiscard]] inline bool passPairMergeCut(PairSep const& s) const
  {
    if (!pairsep.cfgDoPairMergeCut.value) {
      return true;
    }
    if (s.nCommon <= 0) { // nothing was propagated
      return true;
    }
    if (s.dPairMin < pairsep.cfgDPairCut.value) {
      return false;
    }
    const int nCut = pairsep.cfgNCloseCut.value;
    return nCut <= 0 || s.nCloseScaled < nCut;
  }

  [[nodiscard]] AltPairing evaluateAltPairing(PhotonWithLegs const& pPos, PhotonWithLegs const& pEle) const
  {
    return evaluateAltPairingFitter(legOf(pPos, 0), legOf(pEle, 1), mBzT);
  }

  [[nodiscard]] inline bool altIsPhotonLike(AltPairing const& r) const
  {
    return r.fitted && r.dca < crosspair.cfgAltMaxDca.value &&
           r.rxy > crosspair.cfgAltMinR.value && r.rxy < crosspair.cfgAltMaxR.value &&
           r.cospa > crosspair.cfgAltMinCosPA.value && r.mee < crosspair.cfgAltMaxMee.value &&
           r.dcaxy < crosspair.cfgAltMaxDcaXY.value && r.dcaz < crosspair.cfgAltMaxDcaZ.value;
  }

  [[nodiscard]] CrossObs computeCrossObs(PhotonWithLegs const& a, PhotonWithLegs const& b, float qinv) const
  {
    constexpr float kMe = 0.000510999f; // electron mass, GeV/c^2
    CrossObs c;
    if (crosspair.cfgDoCrossPairCut.value && qinv < crosspair.cfgAltMaxQinv.value) {
      c.alt[0] = evaluateAltPairing(a, b); // a's e+ with b's e-
      c.alt[1] = evaluateAltPairing(b, a); // b's e+ with a's e-
      c.nAltValid = (altIsPhotonLike(c.alt[0]) ? 1 : 0) + (altIsPhotonLike(c.alt[1]) ? 1 : 0);
      c.maxAltDca = std::max(c.alt[0].dca, c.alt[1].dca);
      c.maxAltDcaXY = std::max(c.alt[0].dcaxy, c.alt[1].dcaxy);
      c.maxAltDcaZ = std::max(c.alt[0].dcaz, c.alt[1].dcaz);
    }
    auto legVec = [](PhotonWithLegs const& p, int i) {
      return ROOT::Math::PtEtaPhiMVector(p.fLegPt[i], p.fLegEta[i], p.fLegPhi[i], kMe);
    };
    // Combo 0: a's e+ with b's e-.  Combo 1: b's e+ with a's e-.
    for (int combo = 0; combo < 2; ++combo) { // o2-linter: disable=magic-number (the two leg-swap combinations)
      PhotonWithLegs const& pPos = (combo == 0) ? a : b;
      PhotonWithLegs const& pEle = (combo == 0) ? b : a;
      c.mee[combo] = static_cast<float>((legVec(pPos, 0) + legVec(pEle, 1)).M());
    }
    if (qinv > 0.f) {
      c.meeOverQ = std::min(c.mee[0], c.mee[1]) / qinv;
    }
    c.ownMee[0] = static_cast<float>((legVec(a, 0) + legVec(a, 1)).M());
    c.ownMee[1] = static_cast<float>((legVec(b, 0) + legVec(b, 1)).M());
    c.ownMeeMax = std::max(c.ownMee[0], c.ownMee[1]);
    c.deltaMee = c.ownMeeMax - std::max(c.mee[0], c.mee[1]);
    c.m4 = static_cast<float>((legVec(a, 0) + legVec(a, 1) + legVec(b, 0) + legVec(b, 1)).M());
    return c;
  }

  [[nodiscard]] inline bool passCrossPairVeto(CrossObs const& c) const
  {
    if (!crosspair.cfgDoCrossPairCut.value) {
      return true;
    }
    const int mode = crosspair.cfgCrossVetoMode.value;
    const bool passGeom = [&]() {
      const int need = crosspair.cfgAltRequireBoth.value ? 2 : 1;
      return c.nAltValid < need;
    }();
    const bool passDca = (crosspair.cfgCrossMinMaxAltDca.value <= 0.f) ||
                         (c.maxAltDca > crosspair.cfgCrossMinMaxAltDca.value);
    if (mode == 1) { // o2-linter: disable=magic-number (veto mode, see cfgCrossVetoMode)
      return passGeom;
    }
    if (mode == 2) { // o2-linter: disable=magic-number (veto mode, see cfgCrossVetoMode)
      return passDca;
    }
    if (mode == 3) { // o2-linter: disable=magic-number (veto mode, see cfgCrossVetoMode)
      return passGeom && passDca;
    }
    return c.meeOverQ > crosspair.cfgCrossMaxMeeRatio.value;
  }

  template <int ev_id>
  inline void fillCrossPair(bool vetoed)
  {
    constexpr auto dir = (ev_id == 0) ? "Pair/same/CrossPair/" : "Pair/mix/CrossPair/";
    fRegistryCF.fill(HIST(dir) + HIST("hVetoCounter"), vetoed ? 1.0 : 0.0);
  }

  template <typename TLeg>
  static void setDedupLegs(DedupCand& c, TLeg const& pos, TLeg const& ele)
  {
    c.legPt = {static_cast<float>(pos.pt()), static_cast<float>(ele.pt())};
    c.legEta = {static_cast<float>(pos.eta()), static_cast<float>(ele.eta())};
    c.legPhi = {static_cast<float>(pos.phi()), static_cast<float>(ele.phi())};
    c.legHasTpc = {pos.hasTPC(), ele.hasTPC()};
    c.legDeDx = {static_cast<float>(pos.tpcSignal()), static_cast<float>(ele.tpcSignal())};
    c.legFracShared = {static_cast<float>(pos.tpcFractionSharedCls()), static_cast<float>(ele.tpcFractionSharedCls())};
    c.legTrackId = {pos.trackId(), ele.trackId()};
  }

  [[nodiscard]] static LegDistance legDistance(DedupCand const& a, DedupCand const& b, int il)
  {
    LegDistance d;
    d.dEta = std::fabs(a.legEta[il] - b.legEta[il]);
    d.dPhi = std::fabs(RecoDecay::constrainAngle(a.legPhi[il] - b.legPhi[il], -o2::constants::math::PI));
    const float ptSum = a.legPt[il] + b.legPt[il];
    d.ptAsym = (ptSum > kMinSigma) ? std::fabs(a.legPt[il] - b.legPt[il]) / ptSum : 1.f;
    d.dedxValid = a.legHasTpc[il] && b.legHasTpc[il];
    const float dedxSum = a.legDeDx[il] + b.legDeDx[il];
    d.dedxAsym = (d.dedxValid && dedxSum > kMinSigma) ? std::fabs(a.legDeDx[il] - b.legDeDx[il]) / dedxSum : 1.f;
    d.fShared = std::max(a.legFracShared[il], b.legFracShared[il]);
    return d;
  }

  [[nodiscard]] bool legsIdentical(DedupCand const& a,
                                   DedupCand const& b,
                                   int il) const
  {
    if (a.legTrackId[il] == b.legTrackId[il]) {
      return false;
    }

    const auto d = legDistance(a, b, il);

    if (d.dEta > dedup.cfgDupMaxLegDEta.value ||
        d.dPhi > dedup.cfgDupMaxLegDPhi.value) {
      return false;
    }

    if (d.ptAsym > dedup.cfgDupMaxLegPtAsym.value) {
      return false;
    }

    if (d.dedxValid &&
        d.dedxAsym > dedup.cfgDupMaxLegDeDxAsym.value) {
      return false;
    }

    return true;
  }

  [[nodiscard]] bool isDuplicatePair(DedupCand const& a, DedupCand const& b) const
  {
    const float dVtx = std::hypot(a.vx - b.vx, a.vy - b.vy, a.vz - b.vz);
    if (dVtx > dedup.cfgDupMaxDVtx3D.value) {
      return false;
    }
    const bool posSame = legsIdentical(a, b, 0);
    const bool negSame = legsIdentical(a, b, 1);
    return dedup.cfgDupRequireBothLegs.value ? (posSame && negSame) : (posSame || negSame);
  }

  [[nodiscard]] static bool isBetterCand(DedupCand const& a, DedupCand const& b)
  {
    if (a.nITSTPC != b.nITSTPC) {
      return a.nITSTPC > b.nITSTPC;
    }
    return a.chi2 < b.chi2;
  }

  [[nodiscard]] std::unordered_set<int64_t> runDedup(std::vector<DedupCand> const& cands) const
  {
    std::unordered_set<int64_t> rejected;
    if (!dedup.cfgDoDedup.value) {
      return rejected;
    }
    const auto n = static_cast<int>(cands.size());
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        const auto& a = cands[i];
        const auto& b = cands[j];
        if (isDuplicatePair(a, b)) {
          rejected.insert(isBetterCand(a, b) ? b.gi : a.gi);
        }
      }
    }
    return rejected;
  }

  /*************************************************/
  // FILL HELPERS
  /*************************************************/

  template <bool IsMix>
  inline void fillPairEff(TruthGamma const& a, TruthGamma const& b)
  {
    if (!passAsymmetryCut(a.pt, b.pt)) {
      return;
    }
    if (!pairutil::pairBelowQmax(a.pt, a.eta, a.phi, b.pt, b.eta, b.phi, mctruth.cfgMCMaxQinv.value)) {
      return;
    }
    const pairutil::PairQ q = pairutil::computePairQ(ROOT::Math::PtEtaPhiMVector(a.pt, a.eta, a.phi, 0.f),
                                                     ROOT::Math::PtEtaPhiMVector(b.pt, b.eta, b.phi, 0.f));
    if constexpr (!IsMix) {
      fRegistryTruthMC.fill(HIST("MC/PairEff/same/hSparse_Qout_Qinv_Kt_converted"), q.qout, q.qinv, q.kt);
      if (a.legsInV0 && b.legsInV0) {
        fRegistryTruthMC.fill(HIST("MC/PairEff/same/hSparse_Qout_Qinv_Kt_legsInV0"), q.qout, q.qinv, q.kt);
      }
      if (a.v0Built && b.v0Built) {
        fRegistryTruthMC.fill(HIST("MC/PairEff/same/hSparse_Qout_Qinv_Kt_v0Built"), q.qout, q.qinv, q.kt);
      }
      if (a.v0Selected && b.v0Selected) {
        fRegistryTruthMC.fill(HIST("MC/PairEff/same/hSparse_Qout_Qinv_Kt_v0Selected"), q.qout, q.qinv, q.kt);
      }
      if (q.qinv < 0.1f && a.rTrue >= 0.f && b.rTrue >= 0.f) { // o2-linter: disable=magic-number (axis range)
        const float rMean = 0.5f * (a.rTrue + b.rTrue);
        const float dConv = std::min(std::hypot(a.vxTrue - b.vxTrue, a.vyTrue - b.vyTrue, a.vzTrue - b.vzTrue), 19.99f); // o2-linter: disable=magic-number (clamp to the axis)
        fRegistryTruthMC.fill(HIST("MC/PairEff/same/hSparse_Qtrue_Rmean_Dconv_converted"), q.qinv, rMean, dConv);
        if (a.legsInV0 && b.legsInV0) {
          fRegistryTruthMC.fill(HIST("MC/PairEff/same/hSparse_Qtrue_Rmean_Dconv_legsInV0"), q.qinv, rMean, dConv);
        }
        if (a.v0Built && b.v0Built) {
          fRegistryTruthMC.fill(HIST("MC/PairEff/same/hSparse_Qtrue_Rmean_Dconv_v0Built"), q.qinv, rMean, dConv);
        }
        if (a.v0Selected && b.v0Selected) {
          fRegistryTruthMC.fill(HIST("MC/PairEff/same/hSparse_Qtrue_Rmean_Dconv_v0Selected"), q.qinv, rMean, dConv);
        }
      }
    } else {
      fRegistryTruthMC.fill(HIST("MC/PairEff/mix/hSparse_Qout_Qinv_Kt_converted"), q.qout, q.qinv, q.kt);
      if (a.legsInV0 && b.legsInV0) {
        fRegistryTruthMC.fill(HIST("MC/PairEff/mix/hSparse_Qout_Qinv_Kt_legsInV0"), q.qout, q.qinv, q.kt);
      }
      if (a.v0Built && b.v0Built) {
        fRegistryTruthMC.fill(HIST("MC/PairEff/mix/hSparse_Qout_Qinv_Kt_v0Built"), q.qout, q.qinv, q.kt);
      }
      if (a.v0Selected && b.v0Selected) {
        fRegistryTruthMC.fill(HIST("MC/PairEff/mix/hSparse_Qout_Qinv_Kt_v0Selected"), q.qout, q.qinv, q.kt);
      }
    }
  }

  template <int ev_id, typename TCollision>
  void fillPairHistogram(TCollision const& collision,
                         ROOT::Math::PtEtaPhiMVector const& v1,
                         ROOT::Math::PtEtaPhiMVector const& v2,
                         float weight = 1.f)
  {

    const float multNTracks = collision.multNTracksPVeta1();
    const float multFT0M = collision.multFT0M();
    const pairutil::PairQ q = pairutil::computePairQ(v1, v2);
    const float kt = q.kt;
    const float qinv = q.qinv;
    const float qabs_lcms = q.qabsLcms;
    const float qout_lcms = q.qout;
    const float qside_lcms = q.qside;
    const float qlong_lcms = q.qlong;
    if (hbtanalysis.cfgDo3D) {
      if constexpr (ev_id == 0) {
        fRegistryCF.fill(HIST("Pair/same/CF_3D"), std::fabs(qout_lcms), std::fabs(qside_lcms), std::fabs(qlong_lcms), kt, weight);
        if (hbtanalysis.cfgDo2D) {
          fRegistryCF.fill(HIST("Pair/same/CF_2D"), std::fabs(qout_lcms), std::fabs(qinv), multNTracks, multFT0M, kt, weight);
          if (hbtanalysis.cfgDo2DSideLong) {
            fRegistryCF.fill(HIST("Pair/same/CF_2D_Side"), std::fabs(qside_lcms), std::fabs(qinv), multNTracks, multFT0M, kt, weight);
            fRegistryCF.fill(HIST("Pair/same/CF_2D_Long"), std::fabs(qlong_lcms), std::fabs(qinv), multNTracks, multFT0M, kt, weight);
          }
        }
        if (hbtanalysis.cfgDoQinvGate3D) {
          fRegistryCF.fill(HIST("Pair/same/CF_3D_Qinv"), std::fabs(qout_lcms), std::fabs(qside_lcms),
                           std::fabs(qlong_lcms), kt, std::fabs(qinv), weight);
        }
      } else {
        fRegistryCF.fill(HIST("Pair/mix/CF_3D"), std::fabs(qout_lcms), std::fabs(qside_lcms), std::fabs(qlong_lcms), kt, weight);
        if (hbtanalysis.cfgDo2D) {
          fRegistryCF.fill(HIST("Pair/mix/CF_2D"), std::fabs(qout_lcms), std::fabs(qinv), multNTracks, multFT0M, kt, weight);
          if (hbtanalysis.cfgDo2DSideLong) {
            fRegistryCF.fill(HIST("Pair/mix/CF_2D_Side"), std::fabs(qside_lcms), std::fabs(qinv), multNTracks, multFT0M, kt, weight);
            fRegistryCF.fill(HIST("Pair/mix/CF_2D_Long"), std::fabs(qlong_lcms), std::fabs(qinv), multNTracks, multFT0M, kt, weight);
          }
        }
        if (hbtanalysis.cfgDoQinvGate3D) {
          fRegistryCF.fill(HIST("Pair/mix/CF_3D_Qinv"), std::fabs(qout_lcms), std::fabs(qside_lcms),
                           std::fabs(qlong_lcms), kt, std::fabs(qinv), weight);
        }
      }
    } else {
      if constexpr (ev_id == 0) {
        fRegistryCF.fill(HIST("Pair/same/CF_1D"), hbtanalysis.cfgUseLCMS ? qabs_lcms : qinv, kt, weight);
      } else {
        fRegistryCF.fill(HIST("Pair/mix/CF_1D"), hbtanalysis.cfgUseLCMS ? qabs_lcms : qinv, kt, weight);
      }
    }
    if constexpr (ev_id == 0) {
      fRegistryCF.fill(HIST("Pair/same/CF_QLcms_Qinv"), qabs_lcms, std::fabs(qinv), kt, weight);
    } else {
      fRegistryCF.fill(HIST("Pair/mix/CF_QLcms_Qinv"), qabs_lcms, std::fabs(qinv), kt, weight);
    }
    float dEtaPair = v1.Eta() - v2.Eta();
    float dPhiPair = v1.Phi() - v2.Phi();
    dPhiPair = RecoDecay::constrainAngle(dPhiPair, -o2::constants::math::PI);
    if constexpr (ev_id == 0) {
      fRegistryCF.fill(HIST("Pair/same/hSparse_DEtaDPhi_qinv_kT"), dEtaPair, dPhiPair, qinv, kt, weight);
    } else {
      fRegistryCF.fill(HIST("Pair/mix/hSparse_DEtaDPhi_qinv_kT"), dEtaPair, dPhiPair, qinv, kt, weight);
    }
  }

  template <int ev_id, PairTruthType TruthT, typename TCollision>
  void fillPairHistogramMC(TCollision const& collision,
                           ROOT::Math::PtEtaPhiMVector const& v1,
                           ROOT::Math::PtEtaPhiMVector const& v2,
                           float weight = 1.f)
  {
    const float multNTracks = collision.multNTracksPVeta1();
    const float multFT0M = collision.multFT0M();
    const pairutil::PairQ q = pairutil::computePairQ(v1, v2);
    const float kt = q.kt;
    const float qinv = q.qinv;
    const float qabs_lcms = q.qabsLcms;
    const float qout_lcms = q.qout;
    const float qside_lcms = q.qside;
    const float qlong_lcms = q.qlong;
    constexpr auto mcDir = mcDirPrefix<TruthT, ev_id == 1>();
    if (hbtanalysis.cfgDo3D) {
      fRegistryPairMC.fill(HIST(mcDir) + HIST("CF_3D"),
                           std::fabs(qout_lcms), std::fabs(qside_lcms), std::fabs(qlong_lcms), kt, weight);
      if (hbtanalysis.cfgDo2D) {
        fRegistryPairMC.fill(HIST(mcDir) + HIST("CF_2D"), std::fabs(qout_lcms), std::fabs(qinv), multNTracks, multFT0M, kt, weight);
      }
    } else {
      fRegistryPairMC.fill(HIST(mcDir) + HIST("CF_1D"), hbtanalysis.cfgUseLCMS ? qabs_lcms : qinv, kt, weight);
    }
    fRegistryPairMC.fill(HIST(mcDir) + HIST("CF_QLcms_Qinv"), qabs_lcms, std::fabs(qinv), kt, weight);
  }

  template <typename TCollision>
  void fillPairHistogramNoLabel(TCollision const& collision,
                                ROOT::Math::PtEtaPhiMVector const& v1,
                                ROOT::Math::PtEtaPhiMVector const& v2)
  {
    const float multNTracks = collision.multNTracksPVeta1();
    const float multFT0M = collision.multFT0M();
    const pairutil::PairQ q = pairutil::computePairQ(v1, v2);
    const float kt = q.kt;
    const float qinv = q.qinv;
    const float qabs_lcms = q.qabsLcms;
    const float qout_lcms = q.qout;
    const float qside_lcms = q.qside;
    const float qlong_lcms = q.qlong;
    if (hbtanalysis.cfgDo3D) {
      fRegistryPairMC.fill(HIST("Pair/same/MC/NoLabel/CF_3D"), std::fabs(qout_lcms), std::fabs(qside_lcms), std::fabs(qlong_lcms), kt);
      if (hbtanalysis.cfgDo2D) {
        fRegistryPairMC.fill(HIST("Pair/same/MC/NoLabel/CF_2D"), std::fabs(qout_lcms), std::fabs(qinv), multNTracks, multFT0M, kt);
      }
    } else {
      fRegistryPairMC.fill(HIST("Pair/same/MC/NoLabel/CF_1D"), hbtanalysis.cfgUseLCMS ? qabs_lcms : qinv, kt);
    }
  }

  /*************************************************/
  // BUILDERS
  /*************************************************/

  template <typename TG1, typename TG2>
  PairQAObservables buildPairQAObservables(TG1 const& g1, TG2 const& g2)
  {
    PairQAObservables o{};
    o.x1 = g1.vx();
    o.y1 = g1.vy();
    o.z1 = g1.vz();
    o.x2 = g2.vx();
    o.y2 = g2.vy();
    o.z2 = g2.vz();
    o.r1 = std::sqrt(o.x1 * o.x1 + o.y1 * o.y1);
    o.r2 = std::sqrt(o.x2 * o.x2 + o.y2 * o.y2);
    o.dx = o.x1 - o.x2;
    o.dy = o.y1 - o.y2;
    o.dz = o.z1 - o.z2;
    o.deltaR = std::fabs(o.r1 - o.r2);
    o.deltaZ = o.dz;
    o.deltaRxy = std::sqrt(o.dx * o.dx + o.dy * o.dy);
    o.deltaR3D = std::sqrt(o.dx * o.dx + o.dy * o.dy + o.dz * o.dz);
    ROOT::Math::XYZVector cp1(o.x1, o.y1, o.z1), cp2(o.x2, o.y2, o.z2);
    const float mag1 = std::sqrt(cp1.Mag2()), mag2 = std::sqrt(cp2.Mag2());
    if (mag1 < kMinMagnitude || mag2 < kMinMagnitude) {
      o.valid = false;
      return o;
    }
    auto cosPA = static_cast<float>(cp1.Dot(cp2) / (mag1 * mag2));
    cosPA = std::clamp(cosPA, -1.f, 1.f);
    o.opa = std::acos(cosPA);
    o2::math_utils::bringTo02Pi(o.opa);
    if (o.opa > o2::constants::math::PI) {
      o.opa = o2::constants::math::TwoPI - o.opa;
    }
    o.cosOA = std::cos(o.opa / 2.f);
    o.drOverCosOA = (std::fabs(o.cosOA) < kMinCosine) ? 1e12f : (o.deltaR3D / o.cosOA);
    o.v1 = ROOT::Math::PtEtaPhiMVector(g1.pt(), g1.eta(), g1.phi(), 0.f);
    o.v2 = ROOT::Math::PtEtaPhiMVector(g2.pt(), g2.eta(), g2.phi(), 0.f);
    o.k12 = 0.5f * (o.v1 + o.v2);
    o.deta = g1.eta() - g2.eta();
    o.dphi = RecoDecay::constrainAngle(g1.phi() - g2.phi(), -o2::constants::math::PI);
    o.pairEta = 0.5f * (g1.eta() + g2.eta());
    o.pairPhi = RecoDecay::constrainAngle(o.k12.Phi(), 0.f);
    o.kt = o.k12.Pt();
    o.qinv = std::fabs((o.v1 - o.v2).M());
    o.cosTheta = std::fabs(computeCosTheta(o.v1, o.v2));
    o.openingAngle = o.opa;
    return o;
  }

  template <typename TPhoton, typename TLegs, typename TMCParticles>
  static PhotonMCInfo buildPhotonMCInfo(TPhoton const& g, TMCParticles const& mcParticles)
  {
    PhotonMCInfo info{};
    const auto pos = g.template posTrack_as<TLegs>();
    const auto neg = g.template negTrack_as<TLegs>();
    if (pos.has_emmcparticle()) {
      info.mcPosId = pos.emmcparticleId();
    }
    if (neg.has_emmcparticle()) {
      info.mcNegId = neg.emmcparticleId();
    }
    if (!pos.has_emmcparticle() || !neg.has_emmcparticle()) {
      return info;
    }
    info.hasMC = true;
    const auto mcPos = pos.template emmcparticle_as<TMCParticles>();
    const auto mcNeg = neg.template emmcparticle_as<TMCParticles>();
    const int mothIdPos = mcPos.has_mothers() ? mcPos.mothersIds()[0] : -1;
    const int mothIdNeg = mcNeg.has_mothers() ? mcNeg.mothersIds()[0] : -1;
    info.posMotherId = mothIdPos;
    info.negMotherId = mothIdNeg;
    info.posMotherIsPhoton = (mothIdPos >= 0) && (mcParticles.iteratorAt(mothIdPos).pdgCode() == kGamma);
    info.negMotherIsPhoton = (mothIdNeg >= 0) && (mcParticles.iteratorAt(mothIdNeg).pdgCode() == kGamma);
    if (mothIdPos < 0 || mothIdPos != mothIdNeg) {
      return info;
    }
    info.sameMother = true;
    info.motherId = mothIdPos;
    const auto mother = mcParticles.iteratorAt(mothIdPos);
    info.motherPdg = mother.pdgCode();
    info.isTruePhoton = (info.motherPdg == kGamma);
    info.isPhysicalPrimary = mother.isPhysicalPrimary();
    return info;
  }

  static PairTruthType classifyPairTruth(PhotonMCInfo const& m1,
                                         PhotonMCInfo const& m2)
  {
    const bool t1 = m1.hasMC && m1.sameMother && m1.isTruePhoton;
    const bool t2 = m2.hasMC && m2.sameMother && m2.isTruePhoton;

    if (t1 && t2 &&
        m1.motherId >= 0 &&
        m1.motherId == m2.motherId) {
      return PairTruthType::TrueTrueSamePhoton;
    }

    {
      const bool sharedLeg =
        (m1.mcPosId >= 0 &&
         (m1.mcPosId == m2.mcPosId ||
          m1.mcPosId == m2.mcNegId)) ||
        (m1.mcNegId >= 0 &&
         (m1.mcNegId == m2.mcPosId ||
          m1.mcNegId == m2.mcNegId));

      if (sharedLeg) {
        return PairTruthType::SharedMcLeg;
      }
    }

    if (!t1 && !t2) {
      return PairTruthType::FakeFake;
    }

    if (t1 != t2) {
      return PairTruthType::TrueFake;
    }

    return PairTruthType::TrueTrueDistinct;
  }

  template <typename TMCParticles>
  static bool isPi0DaughterPair(PhotonMCInfo const& m1, PhotonMCInfo const& m2,
                                TMCParticles const& mcParticles)
  {
    if (!m1.isTruePhoton || !m2.isTruePhoton || m1.motherId < 0 || m2.motherId < 0) {
      return false;
    }
    const auto ph1 = mcParticles.iteratorAt(m1.motherId);
    const auto ph2 = mcParticles.iteratorAt(m2.motherId);
    if (!ph1.has_mothers() || !ph2.has_mothers()) {
      return false;
    }
    const int gm1 = ph1.mothersIds()[0], gm2 = ph2.mothersIds()[0];
    if (gm1 != gm2) {
      return false;
    }
    return (std::abs(mcParticles.iteratorAt(gm1).pdgCode()) == kPi0);
  }

  template <typename TA, typename TB>
  [[nodiscard]] bool passPairCuts(PairQAObservables const& obs, PairSep const& sep, TA const& a, TB const& b,
                                  pairutil::V0PhotonLegCounts const& ca, pairutil::V0PhotonLegCounts const& cb) const
  {
    if (!passPhotonClassPairCut(ca, cb)) {
      return false;
    }
    if (obs.drOverCosOA < ggpaircuts.cfgMinDRCosOA.value) {
      return false;
    }
    if (!passPairMergeCut(sep)) {
      return false;
    }
    if (!passRZCut(obs.deltaR, obs.deltaZ)) {
      return false;
    }
    if (isInsideEllipse(obs.deta, obs.dphi)) {
      return false;
    }
    return std::max(std::fabs(a.fDcaZToPV), std::fabs(b.fDcaZToPV)) < ggpaircuts.cfgMaxDcaZToPV.value &&
           std::max(std::fabs(a.fDcaXYToPV), std::fabs(b.fDcaXYToPV)) < ggpaircuts.cfgMaxDcaXYToPV.value;
  }

  [[nodiscard]] CrossObs crossObsOfSwapped(PhotonWithLegs const& a, PhotonWithLegs const& b) const
  {
    CrossObs c;
    c.nAltValid = (a.fAnaPhotonLike ? 1 : 0) + (b.fAnaPhotonLike ? 1 : 0);
    c.maxAltDca = std::max(a.fAnaPca, b.fAnaPca);
    c.maxAltDcaXY = std::max(a.fAnaDcaXY, b.fAnaDcaXY);
    c.maxAltDcaZ = std::max(a.fAnaDcaZ, b.fAnaDcaZ);
    c.meeOverQ = 999.f;
    return c;
  }

  template <typename TPhotons, typename TLegs, typename TCut, typename TTracks = std::nullptr_t>
  void collectPhotons(TPhotons const& photonsColl, TCut const& cut, float vtxZ, std::vector<PhotonWithLegs>& pwls, TTracks const* tracks = nullptr)
  {
    std::vector<DedupCand> dedupCands;
    dedupCands.reserve(photonsColl.size());
    for (const auto& g : photonsColl) {
      if (!cut.template IsSelected<decltype(g), TLegs>(g)) {
        continue;
      }
      DedupCand c;
      c.gi = g.globalIndex();
      c.eta = g.eta();
      c.phi = g.phi();
      c.pt = g.pt();
      c.rConv = std::hypot(g.vx(), g.vy());
      c.chi2 = g.chiSquareNDF();
      c.vx = g.vx();
      c.vy = g.vy();
      c.vz = g.vz();
      {
        const auto pos = g.template posTrack_as<TLegs>();
        const auto ele = g.template negTrack_as<TLegs>();
        c.nITSTPC = pairutil::getV0PhotonLegCounts(pos, ele).nITSTPC;
        setDedupLegs(c, pos, ele);
      }
      dedupCands.push_back(c);
    }
    const auto dedupRejected = runDedup(dedupCands);

    pwls.clear();
    pwls.reserve(dedupCands.size());
    for (const auto& g : photonsColl) {
      if (!cut.template IsSelected<decltype(g), TLegs>(g)) {
        continue;
      }
      if (dedup.cfgDoDedup.value && dedupRejected.contains(g.globalIndex())) {
        continue;
      }
      pwls.push_back(makePhotonWithLegs(g, g.template posTrack_as<TLegs>(), g.template negTrack_as<TLegs>(), 0.f, 0.f, vtxZ));
      if (pwls.back().fHasLegXYZ) {
        const auto& q = pwls.back().fLegXYZ;
        const float dLegs = std::hypot(q[0][0] - q[1][0], q[0][1] - q[1][1], q[0][2] - q[1][2]);
        fRegistryCF.fill(HIST("Pair/hLegXYZ_dLegs"), std::min(dLegs, 99.9f));                                              // o2-linter: disable=magic-number (clamp to the axis)
        fRegistryCF.fill(HIST("Pair/hLegXYZ_dLegs_vs_Pca"), std::min(pwls.back().fAnaPca, 4.99f), std::min(dLegs, 4.99f)); // o2-linter: disable=magic-number (clamp to the axis)
        for (const auto& x : q) {
          fRegistryCF.fill(HIST("Pair/hLegXYZ_R_vs_Rconv"), std::hypot(pwls.back().fVx, pwls.back().fVy), std::hypot(x[0], x[1]));
          fRegistryCF.fill(HIST("Pair/hLegXYZ_x_vs_y"), x[0], x[1]);
        }
        {
          const auto& p = pwls.back();
          const AnalyticV0 own = buildAnalyticV0(legOfAtPoint(p, 0), legOfAtPoint(p, 1), 10.f * mBzT, mixing.cfgScoreWeight.value); // o2-linter: disable=magic-number (Tesla -> kGauss)
          fRegistryCF.fill(HIST("Pair/hLegXYZ_RebuildOk"), own.ok ? 1.f : 0.f);
          if (own.ok) {
            fRegistryCF.fill(HIST("Pair/hLegXYZ_PcaRebuilt_vs_Pca"), std::min(p.fAnaPca, 4.99f), std::min(own.pca, 4.99f));              // o2-linter: disable=magic-number (clamp to the axis)
            fRegistryCF.fill(HIST("Pair/hLegXYZ_RRebuilt_vs_R"), std::min(std::hypot(p.fVx, p.fVy), 99.9f), std::min(own.rxy(), 99.9f)); // o2-linter: disable=magic-number (clamp to the axis)
          }
        }
      }
      if constexpr (!std::is_same_v<TTracks, std::nullptr_t>) {
        if (tracks != nullptr) {
          auto& p = pwls.back();
          const auto pos = g.template posTrack_as<TLegs>();
          const auto ele = g.template negTrack_as<TLegs>();
          p.fLegTrack[0] = getTrackParCov(tracks->rawIteratorAt(static_cast<uint64_t>(pos.trackId())));
          p.fLegTrack[1] = getTrackParCov(tracks->rawIteratorAt(static_cast<uint64_t>(ele.trackId())));
          p.fHasTrackCov = true;
        }
      }
    }
  }

  struct LegBuilt {
    AnalyticV0 v;
    size_t iPos{0}, iNeg{0};
    int64_t posTrack{-1}, negTrack{-1};
  };
  template <typename TCollision>
  void runLegBuiltReference(TCollision const& collision, std::vector<PhotonWithLegs> const& evA, std::vector<PhotonWithLegs> const& evB)
  {
    const float w = mixing.cfgScoreWeight.value;
    const float bzkG = 10.f * mBzT; // o2-linter: disable=magic-number (Tesla -> kGauss)
    auto accept = [&](AnalyticV0 const& v) {
      return v.ok && v.pca < mixbuilder.cfgBuilderMaxPca.value && v.rxy() > mixbuilder.cfgBuilderMinR.value && v.rxy() < mixbuilder.cfgBuilderMaxR.value;
    };
    std::vector<LegBuilt> gAB, gBA;
    for (size_t ia = 0; ia < evA.size(); ++ia) {
      for (size_t ib = 0; ib < evB.size(); ++ib) {
        const auto& a = evA[ia];
        const auto& b = evB[ib];
        if (!pairutil::pairBelowQmax(a.pt(), a.eta(), a.phi(), b.pt(), b.eta(), b.phi(), mixing.cfgQuartetMaxQinv.value)) {
          continue;
        }
        const AnalyticV0 x1 = buildAnalyticV0(legOf(a, 0), legOf(b, 1), bzkG, w); // e+(A) e-(B)
        if (accept(x1)) {
          gAB.push_back({x1, ia, ib, a.fLegTrackId[0], b.fLegTrackId[1]});
        }
        const AnalyticV0 x2 = buildAnalyticV0(legOf(b, 0), legOf(a, 1), bzkG, w); // e+(B) e-(A)
        if (accept(x2)) {
          gBA.push_back({x2, ib, ia, b.fLegTrackId[0], a.fLegTrackId[1]});
        }
      }
    }
    auto unique = [](std::vector<LegBuilt>& g) {
      std::sort(g.begin(), g.end(), [](LegBuilt const& x, LegBuilt const& y) { return x.v.pca < y.v.pca; });
      std::vector<LegBuilt> kept;
      std::unordered_set<int64_t> posUsed, negUsed;
      for (const auto& c : g) {
        if (posUsed.contains(c.posTrack) || negUsed.contains(c.negTrack)) {
          continue;
        }
        posUsed.insert(c.posTrack);
        negUsed.insert(c.negTrack);
        kept.push_back(c);
      }
      g.swap(kept);
    };
    unique(gAB);
    unique(gBA);
    fRegistryCF.fill(HIST("Pair/mix/hLegBuilt_NAB_NBA"), static_cast<float>(std::min(gAB.size(), static_cast<size_t>(29))), static_cast<float>(std::min(gBA.size(), static_cast<size_t>(29)))); // o2-linter: disable=magic-number (last bin)
    for (const auto& c : gAB) {
      fRegistryCF.fill(HIST("Pair/mix/hLegBuilt_Pca"), std::min(c.v.pca, 4.99f)); // o2-linter: disable=magic-number (clamp to the axis)
    }
    for (const auto& c : gBA) {
      fRegistryCF.fill(HIST("Pair/mix/hLegBuilt_Pca"), std::min(c.v.pca, 4.99f)); // o2-linter: disable=magic-number (clamp to the axis)
    }
    for (const auto& x : gAB) {
      const CrossPhotonLite px = crossPhotonLite(x.v, evA[x.iPos]);
      for (const auto& y : gBA) {
        const CrossPhotonLite py = crossPhotonLite(y.v, evA[y.iNeg]);
        if (!passAsymmetryCut(px.pt(), py.pt())) {
          continue;
        }
        if (!passFastQinvGate(px, py)) {
          continue;
        }
        auto obs = buildPairQAObservables(px, py);
        if (!obs.valid) {
          continue;
        }
        const bool sameQuartet = (x.iPos == y.iNeg) && (x.iNeg == y.iPos);
        const auto sep = computePairSep(evA[x.iPos], evB[x.iNeg]);
        const auto lcX = legCountsFromMasks(evA[x.iPos].fLegDetMask[0], evB[x.iNeg].fLegDetMask[1]);
        const auto lcY = legCountsFromMasks(evB[y.iPos].fLegDetMask[0], evA[y.iNeg].fLegDetMask[1]);
        if (!passPairCuts(obs, sep, px, py, lcX, lcY)) {
          continue;
        }
        fRegistryCF.fill(HIST("Pair/mix/hLegBuilt_SameQuartet_vs_Qinv"), obs.qinv, sameQuartet ? 1.f : 0.f);
        fillPairHistogram<1>(collision, obs.v1, obs.v2, 1.f);
        const bool labelled = evA[x.iPos].fIsTruePhoton >= 0 && evB[x.iNeg].fIsTruePhoton >= 0 && evB[y.iPos].fIsTruePhoton >= 0 && evA[y.iNeg].fIsTruePhoton >= 0;
        if (labelled) {
          fillPairHistogramMC<1, PairTruthType::FakeFake>(collision, obs.v1, obs.v2);
        }
      }
    }
  }

  template <typename TCollision>
  void runMixedEvent(TCollision const& collision, std::vector<PhotonWithLegs> const& pwls,
                     std::tuple<int, int, int, int> const& keyBin, std::pair<int, int> const& keyDFCollision)
  {
    auto poolIDs = emh1->GetCollisionIdsFromEventPool(keyBin);
    for (const auto& mixID : poolIDs) {
      if (mixID.second == collision.globalIndex() && mixID.first == ndf) {
        continue;
      }
      const uint64_t bcMix = mapMixedEventIdToGlobalBC[mixID];
      const uint64_t diffBC = std::max(collision.globalBC(), bcMix) - std::min(collision.globalBC(), bcMix);
      fRegistryCF.fill(HIST("Pair/mix/hDiffBC"), diffBC);
      if (diffBC < mixing.ndiffBCMix) {
        continue;
      }
      auto poolPhotons = emh1->GetTracksPerCollision(mixID);
      if (mixing.cfgMixMode.value == 2) { // o2-linter: disable=magic-number (mode 2, see cfgMixMode)
        runLegBuiltReference(collision, pwls, poolPhotons);
        continue;
      }
      for (const auto& g1 : pwls) {
        for (const auto& g2 : poolPhotons) {
          if (!passFastQinvGate(g1, g2)) {
            continue;
          }
          auto obs = buildPairQAObservables(g1, g2);
          if (!obs.valid) {
            continue;
          }
          const auto sep = computePairSep(g1, g2);

          Emulation emu;
          if (mixing.cfgMixMode.value == 1) {
            emu = emulateCrossPairing(g1, g2, obs.qinv);
            fRegistryCF.fill(HIST("Pair/mix/hQuartetOutcome"), obs.qinv, static_cast<float>(emu.outcome));
            if (emu.outcome != 0) {
              const float d3D = std::hypot((g1.fVx - g1.fVtxX) - (g2.fVx - g2.fVtxX), (g1.fVy - g1.fVtxY) - (g2.fVy - g2.fVtxY), (g1.fVz - g1.fVtxZ) - (g2.fVz - g2.fVtxZ));
              fRegistryCF.fill(HIST("Pair/mix/hVtxDist_vs_Qinv_Outcome"), obs.qinv, static_cast<float>(emu.outcome), std::min(d3D, 49.99f)); // o2-linter: disable=magic-number (clamp to the axis)
            }
            if (emu.outcome != 0) {
              fillFailMask(fRegistryCF, HIST("Pair/mix/hQuartetCrossFail_vs_Qinv"), emu.failMask[0], obs.qinv);
              fillFailMask(fRegistryCF, HIST("Pair/mix/hQuartetCrossFail_vs_Qinv"), emu.failMask[1], obs.qinv);
            }
            if (emu.crossExists) {
              fRegistryCF.fill(HIST("Pair/mix/hQuartetDeltaScore_vs_Qinv"), obs.qinv, std::clamp(emu.dSBest, -19.99f, 19.99f)); // o2-linter: disable=magic-number (clamp to the axis)
            }
          }
          if (emu.outcome == 3) { // o2-linter: disable=magic-number (pair lost)
          }
          if (emu.swapped) {

            const CrossPhotonLite x1 = crossPhotonLite(emu.x1, g1);
            const CrossPhotonLite x2 = crossPhotonLite(emu.x2, g1);
            auto obsX = buildPairQAObservables(x1, x2);
            if (!obsX.valid) {
              continue;
            }
            fRegistryCF.fill(HIST("Pair/mix/hQuartetQinvCross_vs_Qinv"), obs.qinv, obsX.qinv);
            if (!passAsymmetryCut(x1.pt(), x2.pt())) {
              continue;
            }
            const auto lc1 = legCountsFromMasks(g1.fLegDetMask[0], g2.fLegDetMask[1]);
            const auto lc2 = legCountsFromMasks(g2.fLegDetMask[0], g1.fLegDetMask[1]);
            if (!passPairCuts(obsX, sep, x1, x2, lc1, lc2)) {
              continue;
            }
            if (crosspair.cfgDoCrossPairCut.value) {
              const bool vetoed = !passCrossPairVeto(crossObsOfSwapped(g1, g2));
              fillCrossPair<1>(vetoed);
              if (vetoed) {
                continue;
              }
            }
            fillPairHistogram<1>(collision, obsX.v1, obsX.v2, 1.f);
            if (g1.fIsTruePhoton >= 0 && g2.fIsTruePhoton >= 0) {
              fillPairHistogramMC<1, PairTruthType::FakeFake>(collision, obsX.v1, obsX.v2);
            }
            continue;
          }

          if (!passAsymmetryCut(g1.pt(), g2.pt())) {
            continue;
          }
          if (!passPairCuts(obs, sep, g1, g2, g1.legCounts(), g2.legCounts())) {
            continue;
          }
          if (crosspair.cfgDoCrossPairCut.value) {
            const bool vetoed = !passCrossPairVeto(computeCrossObs(g1, g2, obs.qinv));
            fillCrossPair<1>(vetoed);
            if (vetoed) {
              continue;
            }
          }
          fillPairHistogram<1>(collision, obs.v1, obs.v2, 1.f);
          if (g1.fIsTruePhoton >= 0 && g2.fIsTruePhoton >= 0) {
            const int nTrue = g1.fIsTruePhoton + g2.fIsTruePhoton;
            if (nTrue == 2) { // o2-linter: disable=magic-number (both photons are true)
              fillPairHistogramMC<1, PairTruthType::TrueTrueDistinct>(collision, obs.v1, obs.v2);
            } else if (nTrue == 1) {
              fillPairHistogramMC<1, PairTruthType::TrueFake>(collision, obs.v1, obs.v2);
            } else {
              fillPairHistogramMC<1, PairTruthType::FakeFake>(collision, obs.v1, obs.v2);
            }
          }
        }
      }
    }
    for (const auto& p : pwls) {
      emh1->AddTrackToEventPool(keyDFCollision, p);
    }
    emh1->AddCollisionIdAtLast(keyBin, keyDFCollision);
    emh2->AddCollisionIdAtLast(keyBin, keyDFCollision);
    mapMixedEventIdToGlobalBC[keyDFCollision] = collision.globalBC();
  }

  template <typename TCollisions, typename TPhotons, typename TLegs, typename TPreslice, typename TCut, typename TTracks = std::nullptr_t>
  void runPairing(TCollisions const& collisions, TPhotons const& photons, TLegs const& /*legs*/,
                  TPreslice const& perCollision, TCut const& cut, TTracks const* tracks = nullptr)
  {
    std::vector<PhotonWithLegs> pwls;
    for (const auto& collision : collisions) {
      initCCDB(collision);
      const std::array<float, 3> cent = {collision.centFT0M(), collision.centFT0A(), collision.centFT0C()};
      if (cent[mixing.cfgCentEstimator] < centralitySelection.cfgCentMin || centralitySelection.cfgCentMax < cent[mixing.cfgCentEstimator]) {
        continue;
      }
      const std::array<float, 7> epArr = {collision.ep2ft0m(), collision.ep2ft0a(), collision.ep2ft0c(),
                                          collision.ep2fv0a(), collision.ep2btot(), collision.ep2bpos(), collision.ep2bneg()};
      const float ep2 = epArr[mixing.cfgEP2EstimatorForMix];
      fRegistry.fill(HIST("Event/before/hEP2_CentFT0C_forMix"), collision.centFT0C(), ep2);
      o2::aod::pwgem::photonmeson::utils::eventhistogram::fillEventInfo<0>(&fRegistry, collision, 1.f);
      if (!fEMEventCut.IsSelected(collision)) {
        continue;
      }
      o2::aod::pwgem::photonmeson::utils::eventhistogram::fillEventInfo<1>(&fRegistry, collision, 1.f);
      fRegistry.fill(HIST("Event/before/hCollisionCounter"), 12.0);
      fRegistry.fill(HIST("Event/after/hCollisionCounter"), 12.0);
      fRegistry.fill(HIST("Event/after/hEP2_CentFT0C_forMix"), collision.centFT0C(), ep2);
      const float occupancy = (mixing.cfgOccupancyEstimator == 1)
                                ? static_cast<float>(collision.trackOccupancyInTimeRange())
                                : collision.ft0cOccupancyInTimeRange();
      const int zbin = binOf(ztxBinEdges, collision.posZ()), centbin = binOf(centBinEdges, cent[mixing.cfgCentEstimator]);
      const int epbin = binOf(epBinEgdes, ep2), occbin = binOf(occBinEdges, occupancy);
      const auto keyBin = std::make_tuple(zbin, centbin, epbin, occbin);
      const auto keyDFCollision = std::make_pair(ndf, static_cast<int>(collision.globalIndex()));
      auto photonsColl = photons.sliceBy(perCollision, collision.globalIndex());

      collectPhotons<decltype(photonsColl), TLegs>(photonsColl, cut, collision.posZ(), pwls, tracks);
      fRegistryCF.fill(HIST("Pair/hNPhotonsInPool"), static_cast<float>(std::min(pwls.size(), static_cast<size_t>(20)))); // o2-linter: disable=magic-number (last bin)

      // ─── same event ───
      for (size_t i = 0; i < pwls.size(); ++i) {
        for (size_t j = i + 1; j < pwls.size(); ++j) {
          const auto& pwl1 = pwls[i];
          const auto& pwl2 = pwls[j];
          if (pwl1.sharesTrackWith(pwl2)) {
            continue;
          }
          if (!passAsymmetryCut(pwl1.pt(), pwl2.pt())) {
            continue;
          }
          auto obs = buildPairQAObservables(pwl1, pwl2);
          if (!obs.valid) {
            continue;
          }
          const auto sep = computePairSep(pwl1, pwl2);
          if (!passPairCuts(obs, sep, pwl1, pwl2, pwl1.legCounts(), pwl2.legCounts())) {
            continue;
          }
          if (crosspair.cfgDoCrossPairCut.value) {
            const bool vetoed = !passCrossPairVeto(computeCrossObs(pwl1, pwl2, obs.qinv));
            fillCrossPair<0>(vetoed);
            if (vetoed) {
              continue;
            }
          }
          fillPairHistogram<0>(collision, obs.v1, obs.v2, 1.f);
        }
      }

      // ─── mixed event ───
      if (!mixing.cfgDoMix || pwls.empty()) {
        continue;
      }
      runMixedEvent(collision, pwls, keyBin, keyDFCollision);
    }
  }

  template <soa::is_table TCollisions, soa::is_table TPhotons,
            soa::is_table TLegs, soa::is_table TMCParticles,
            typename TPreslice, typename TCut, typename TTracks = std::nullptr_t>
  void runPairingMC(TCollisions const& collisions, TPhotons const& photons,
                    TLegs const& /*legs*/, TMCParticles const& mcParticles,
                    TPreslice const& perCollision, TCut const& cut, TTracks const* tracks = nullptr)
  {
    std::vector<PhotonWithLegs> pwls;
    std::vector<PhotonMCInfo> mcs;
    for (const auto& collision : collisions) {
      initCCDB(collision);
      const std::array<float, 3> cent = {collision.centFT0M(), collision.centFT0A(), collision.centFT0C()};
      if (cent[mixing.cfgCentEstimator] < centralitySelection.cfgCentMin || centralitySelection.cfgCentMax < cent[mixing.cfgCentEstimator]) {
        continue;
      }
      const std::array<float, 7> epArr = {collision.ep2ft0m(), collision.ep2ft0a(), collision.ep2ft0c(),
                                          collision.ep2fv0a(), collision.ep2btot(), collision.ep2bpos(), collision.ep2bneg()};
      const float ep2 = epArr[mixing.cfgEP2EstimatorForMix];
      fRegistry.fill(HIST("Event/before/hEP2_CentFT0C_forMix"), collision.centFT0C(), ep2);
      o2::aod::pwgem::photonmeson::utils::eventhistogram::fillEventInfo<0>(&fRegistry, collision, 1.f);
      if (!fEMEventCut.IsSelected(collision)) {
        continue;
      }
      o2::aod::pwgem::photonmeson::utils::eventhistogram::fillEventInfo<1>(&fRegistry, collision, 1.f);
      fRegistry.fill(HIST("Event/before/hCollisionCounter"), 12.0);
      fRegistry.fill(HIST("Event/after/hCollisionCounter"), 12.0);
      fRegistry.fill(HIST("Event/after/hEP2_CentFT0C_forMix"), collision.centFT0C(), ep2);
      const float occupancy = (mixing.cfgOccupancyEstimator == 1)
                                ? static_cast<float>(collision.trackOccupancyInTimeRange())
                                : collision.ft0cOccupancyInTimeRange();
      const int zbin = binOf(ztxBinEdges, collision.posZ()), centbin = binOf(centBinEdges, cent[mixing.cfgCentEstimator]);
      const int epbin = binOf(epBinEgdes, ep2), occbin = binOf(occBinEdges, occupancy);
      const auto keyBin = std::make_tuple(zbin, centbin, epbin, occbin);
      const auto keyDFCollision = std::make_pair(ndf, static_cast<int>(collision.globalIndex()));
      auto photonsColl = photons.sliceBy(perCollision, collision.globalIndex());

      collectPhotons<decltype(photonsColl), TLegs>(photonsColl, cut, collision.posZ(), pwls, tracks);
      fRegistryCF.fill(HIST("Pair/hNPhotonsInPool"), static_cast<float>(std::min(pwls.size(), static_cast<size_t>(20)))); // o2-linter: disable=magic-number (last bin)
      mcs.clear();
      mcs.reserve(pwls.size());
      {
        size_t ip = 0;
        for (const auto& g : photonsColl) {
          if (ip < pwls.size() && pwls[ip].fGlobalIndex == g.globalIndex()) {
            mcs.push_back(buildPhotonMCInfo<decltype(g), TLegs>(g, mcParticles));
            pwls[ip].fIsTruePhoton = (mcs.back().sameMother && mcs.back().isTruePhoton) ? 1 : 0;
            ++ip;
          }
        }
      }

      // ─── same event ───
      for (size_t i = 0; i < pwls.size(); ++i) {
        for (size_t j = i + 1; j < pwls.size(); ++j) {
          const auto& pwl1 = pwls[i];
          const auto& pwl2 = pwls[j];
          const auto& mc1 = mcs[i];
          const auto& mc2 = mcs[j];
          if (pwl1.sharesTrackWith(pwl2)) {
            continue;
          }
          auto truthType = classifyPairTruth(mc1, mc2);
          if (truthType == PairTruthType::TrueTrueDistinct && isPi0DaughterPair(mc1, mc2, mcParticles)) {
            truthType = PairTruthType::Pi0Daughters;
          }
          {
            const float qReco = pairutil::computePairQ(pwl1, pwl2).qinv;
            float d3D = std::hypot(pwl1.fVx - pwl2.fVx, pwl1.fVy - pwl2.fVy, pwl1.fVz - pwl2.fVz);
            if (truthType == PairTruthType::FakeFake && mc1.mcPosId >= 0 && mc1.mcNegId >= 0) {
              const auto lA = mcParticles.iteratorAt(mc1.mcPosId);
              const auto lB = mcParticles.iteratorAt(mc1.mcNegId);
              d3D = std::hypot(static_cast<float>(lA.vx() - lB.vx()), static_cast<float>(lA.vy() - lB.vy()), static_cast<float>(lA.vz() - lB.vz()));
            }
            fRegistryPairMC.fill(HIST("Pair/same/MC/hVtxDist_vs_Qinv_Type"), qReco, static_cast<float>(truthType), std::min(d3D, 49.99f)); // o2-linter: disable=magic-number (clamp to the axis)
            if (truthType == PairTruthType::TrueTrueDistinct) {
              fRegistryPairMC.fill(HIST("Pair/same/MC/hTransfer_vs_Qorig"), qReco, 0.f);
            } else if (truthType == PairTruthType::FakeFake && mc1.posMotherIsPhoton && mc1.negMotherIsPhoton &&
                       mc1.posMotherId == mc2.negMotherId && mc1.negMotherId == mc2.posMotherId) {
              const auto phA = mcParticles.iteratorAt(mc1.posMotherId);
              const auto phB = mcParticles.iteratorAt(mc1.negMotherId);
              fRegistryPairMC.fill(HIST("Pair/same/MC/hTransfer_vs_Qorig"), pairutil::computePairQ(phA, phB).qinv, 1.f);
            }
          }
          if (!passAsymmetryCut(pwl1.pt(), pwl2.pt())) {
            continue;
          }
          auto obs = buildPairQAObservables(pwl1, pwl2);
          if (!obs.valid) {
            continue;
          }
          const auto sep = computePairSep(pwl1, pwl2);
          if (obs.drOverCosOA < ggpaircuts.cfgMinDRCosOA.value || !passPairMergeCut(sep) ||
              !passRZCut(obs.deltaR, obs.deltaZ) || isInsideEllipse(obs.deta, obs.dphi) ||
              !passPointingPairCut(pwl1, pwl2)) {
            continue;
          }
          if (crosspair.cfgDoCrossPairCut.value) {
            const bool vetoed = !passCrossPairVeto(computeCrossObs(pwl1, pwl2, obs.qinv));
            fillCrossPair<0>(vetoed);
            if (vetoed) {
              continue;
            }
          }
          fillPairHistogram<0>(collision, obs.v1, obs.v2, 1.f);

          if (!mc1.hasMC || !mc2.hasMC) {
            fillPairHistogramNoLabel(collision, obs.v1, obs.v2);
            continue;
          }
          const auto truthAxis = static_cast<float>(static_cast<int>(truthType));
          fRegistryPairMC.fill(HIST("Pair/same/MC/hTruthTypeVsQinv"), obs.qinv, truthAxis);
          fRegistryPairMC.fill(HIST("Pair/same/MC/hTruthTypeVsKt"), obs.kt, truthAxis);
          {

            const Emulation emu = emulateCrossPairing(pwl1, pwl2, obs.qinv);
            fRegistryPairMC.fill(HIST("Pair/same/MC/hQuartetOutcome_vs_Truth"), obs.qinv, truthAxis, static_cast<float>(emu.outcome));
            if (emu.outcome != 0) {
              fillFailMask(fRegistryPairMC, HIST("Pair/same/MC/hQuartetCrossFail_vs_Qinv"), emu.failMask[0], obs.qinv);
              fillFailMask(fRegistryPairMC, HIST("Pair/same/MC/hQuartetCrossFail_vs_Qinv"), emu.failMask[1], obs.qinv);
            }
            if (emu.crossExists) {
              fRegistryPairMC.fill(HIST("Pair/same/MC/hQuartetDeltaScore_vs_Truth"), obs.qinv, truthAxis, std::clamp(emu.dSBest, -19.99f, 19.99f)); // o2-linter: disable=magic-number (clamp to the axis)
            }
          }
          switch (truthType) {
            case PairTruthType::TrueTrueDistinct:
              fillPairHistogramMC<0, PairTruthType::TrueTrueDistinct>(collision, obs.v1, obs.v2);
              break;
            case PairTruthType::TrueTrueSamePhoton:
              fillPairHistogramMC<0, PairTruthType::TrueTrueSamePhoton>(collision, obs.v1, obs.v2);
              break;
            case PairTruthType::SharedMcLeg:
              fillPairHistogramMC<0, PairTruthType::SharedMcLeg>(collision, obs.v1, obs.v2);
              break;
            case PairTruthType::TrueFake:
              fillPairHistogramMC<0, PairTruthType::TrueFake>(collision, obs.v1, obs.v2);
              break;
            case PairTruthType::FakeFake:
              fillPairHistogramMC<0, PairTruthType::FakeFake>(collision, obs.v1, obs.v2);
              break;
            case PairTruthType::Pi0Daughters:
              fillPairHistogramMC<0, PairTruthType::Pi0Daughters>(collision, obs.v1, obs.v2);
              break;
            default:
              break;
          }
        }
      }

      if (!mixing.cfgDoMix || pwls.empty()) {
        continue;
      }
      runMixedEvent(collision, pwls, keyBin, keyDFCollision);
    }
  }

  template <soa::is_table TCollisions, soa::is_table TPhotons,
            soa::is_table TLegs, soa::is_table TMCParticles,
            soa::is_table TMCEvents,
            typename TPresliceMCParts, typename TPresliceLegs, typename TCut>
  void runTruthEfficiency(TCollisions const& collisions,
                          TPhotons const& v0photons,
                          TLegs const& v0legs,
                          TMCParticles const& emmcParticles,
                          TMCEvents const& /*mcEvents*/,
                          TPresliceMCParts const& perMCCollision,
                          TPresliceLegs const& perCollisionLegs,
                          TCut const& cut)
  {
    auto wrapPhi = [](float dphi) -> float {
      return RecoDecay::constrainAngle(dphi, -o2::constants::math::PI);
    };

    for (const auto& collision : collisions) {
      initCCDB(collision);
      if (!fEMEventCut.IsSelected(collision)) {
        continue;
      }
      const std::array<float, 3> cent = {collision.centFT0M(), collision.centFT0A(), collision.centFT0C()};
      if (cent[mixing.cfgCentEstimator] < centralitySelection.cfgCentMin ||
          centralitySelection.cfgCentMax < cent[mixing.cfgCentEstimator]) {
        continue;
      }
      if (!collision.has_emmcevent()) {
        continue;
      }

      const int64_t thisCollisionId = collision.globalIndex();
      const int mcEventId = collision.template emmcevent_as<TMCEvents>().globalIndex();

      auto recoPhotonsColl = v0photons.sliceBy(perCollisionPCM, thisCollisionId);
      auto emmcPartsColl = emmcParticles.sliceBy(perMCCollision, mcEventId);
      auto legsColl = v0legs.sliceBy(perCollisionLegs, thisCollisionId);

      std::unordered_set<int> legIdsThisCollision;
      legIdsThisCollision.reserve(legsColl.size());
      for (const auto& leg : legsColl) {
        if (leg.has_emmcparticle()) {
          legIdsThisCollision.insert(leg.emmcparticleId());
        }
      }

      struct PhotonRecoInfo {
        bool hasV0 = false, passesCut = false;
      };
      std::unordered_map<int, PhotonRecoInfo> gammaRecoMap;
      gammaRecoMap.reserve(recoPhotonsColl.size());

      for (const auto& g : recoPhotonsColl) {
        const auto pos = g.template posTrack_as<TLegs>();
        const auto neg = g.template negTrack_as<TLegs>();
        if (pos.collisionId() != thisCollisionId || neg.collisionId() != thisCollisionId) {
          continue;
        }
        const auto truth = pmmc::makePhotonMCInfo<TLegs>(g, emmcParticles);
        if (!truth.isTruePhoton) {
          continue;
        }
        auto& info = gammaRecoMap[truth.posPhotonId];
        info.hasV0 = true;
        info.passesCut = info.passesCut || cut.template IsSelected<std::decay_t<decltype(g)>, TLegs>(g);
      }

      // ─── Build true gamma list ────────────────────────────────────────────────
      std::vector<TruthGamma> trueGammas;
      trueGammas.reserve(32);

      for (const auto& g : emmcPartsColl) {
        if (g.pdgCode() != kGamma) {
          continue;
        }
        if (!g.isPhysicalPrimary() && !g.producedByGenerator()) {
          continue;
        }
        if (std::fabs(g.eta()) > pcmcuts.cfgMaxEtaV0.value) {
          continue;
        }
        const float mcV0PtMin = (mctruth.cfgMCMinV0Pt.value > 0.f)
                                  ? mctruth.cfgMCMinV0Pt.value
                                  : pcmcuts.cfgMinPtV0.value;
        if (g.pt() < mcV0PtMin) {
          continue;
        }
        if (!g.has_daughters()) {
          continue;
        }

        int posId = -1, negId = -1;
        float rTrue = -1.f;
        float convX = 0.f, convY = 0.f, convZ = 0.f;
        for (const auto& dId : g.daughtersIds()) {
          if (dId < 0) {
            continue;
          }
          const auto d = emmcParticles.iteratorAt(dId);
          if (d.pdgCode() == kPositron) {
            posId = dId;
            rTrue = std::sqrt(d.vx() * d.vx() + d.vy() * d.vy());
            convX = static_cast<float>(d.vx());
            convY = static_cast<float>(d.vy());
            convZ = static_cast<float>(d.vz());
          } else if (d.pdgCode() == kElectron) {
            negId = dId;
          }
        }
        if (posId < 0 || negId < 0) {
          continue;
        }
        if (rTrue < mctruth.cfgMCMinRconv.value || rTrue > mctruth.cfgMCMaxRconv.value) {
          continue;
        }

        const auto mcPosE = emmcParticles.iteratorAt(posId);
        const auto mcNegE = emmcParticles.iteratorAt(negId);

        if (mctruth.cfgMCMinLegPt.value > 0.f &&
            (static_cast<float>(mcPosE.pt()) < mctruth.cfgMCMinLegPt.value ||
             static_cast<float>(mcNegE.pt()) < mctruth.cfgMCMinLegPt.value)) {
          continue;
        }
        if (std::fabs(static_cast<float>(mcPosE.eta())) > mctruth.cfgMCMaxLegEta.value ||
            std::fabs(static_cast<float>(mcNegE.eta())) > mctruth.cfgMCMaxLegEta.value) {
          continue;
        }

        const auto deTrE = static_cast<float>(mcPosE.eta() - mcNegE.eta());
        const auto dpTrE = wrapPhi(static_cast<float>(mcPosE.phi() - mcNegE.phi()));
        const auto legDRt = std::sqrt(deTrE * deTrE + dpTrE * dpTrE);

        const auto pxG = static_cast<float>(g.px()), pyG = static_cast<float>(g.py()),
                   pzG = static_cast<float>(g.pz());
        const auto magG = std::sqrt(pxG * pxG + pyG * pyG + pzG * pzG);
        float alphaTrue = 0.f;
        if (magG > kMinSigma) {
          const float ux = pxG / magG, uy = pyG / magG, uz = pzG / magG;
          const float pLpos = static_cast<float>(mcPosE.px()) * ux +
                              static_cast<float>(mcPosE.py()) * uy +
                              static_cast<float>(mcPosE.pz()) * uz;
          const float pLneg = static_cast<float>(mcNegE.px()) * ux +
                              static_cast<float>(mcNegE.py()) * uy +
                              static_cast<float>(mcNegE.pz()) * uz;
          const float sumPL = pLpos + pLneg;
          if (std::fabs(sumPL) > kMinSigma) {
            alphaTrue = (pLpos - pLneg) / sumPL;
          }
        }

        TruthGamma tg;
        tg.id = static_cast<int>(g.globalIndex());
        tg.posId = posId;
        tg.negId = negId;
        tg.eta = static_cast<float>(g.eta());
        tg.phi = static_cast<float>(g.phi());
        tg.pt = static_cast<float>(g.pt());
        tg.rTrue = rTrue;
        tg.vxTrue = convX;
        tg.vyTrue = convY;
        tg.vzTrue = convZ;
        tg.legDRtrue = legDRt;
        tg.legDEta = deTrE;
        tg.legDPhi = dpTrE;
        tg.alphaTrue = alphaTrue;
        tg.legPtTrue = {static_cast<float>(mcPosE.pt()), static_cast<float>(mcNegE.pt())};
        tg.legEtaTrue = {static_cast<float>(mcPosE.eta()), static_cast<float>(mcNegE.eta())};
        tg.legPhiTrue = {static_cast<float>(mcPosE.phi()), static_cast<float>(mcNegE.phi())};
        tg.legsInV0 = (legIdsThisCollision.contains(posId) > 0) && (legIdsThisCollision.contains(negId) > 0);
        if (const auto itR = gammaRecoMap.find(tg.id); itR != gammaRecoMap.end()) {
          tg.v0Built = itR->second.hasV0;
          tg.v0Selected = itR->second.passesCut;
        }
        trueGammas.push_back(tg);
      }

      const bool doTruthMix = mctruth.cfgDoTruthMix.value;
      const bool doPairEff = mctruth.cfgDoPairEff.value;
      if (doTruthMix || doPairEff) {
        const float centForBin = cent[mixing.cfgCentEstimator.value];
        const std::array<float, 7> epArr = {collision.ep2ft0m(), collision.ep2ft0a(), collision.ep2ft0c(),
                                            collision.ep2fv0a(), collision.ep2btot(), collision.ep2bpos(),
                                            collision.ep2bneg()};
        const float ep2 = epArr[mixing.cfgEP2EstimatorForMix.value];
        const float occupancy = (mixing.cfgOccupancyEstimator.value == 1)
                                  ? static_cast<float>(collision.trackOccupancyInTimeRange())
                                  : collision.ft0cOccupancyInTimeRange();
        const auto keyBin = std::make_tuple(binOf(ztxBinEdges, collision.posZ()),
                                            binOf(centBinEdges, centForBin),
                                            binOf(epBinEgdes, ep2),
                                            binOf(occBinEdges, occupancy));

        if (doPairEff) {
          for (size_t i = 0; i < trueGammas.size(); ++i) {
            for (size_t j = i + 1; j < trueGammas.size(); ++j) {
              fillPairEff<false>(trueGammas[i], trueGammas[j]);
            }
          }
          if (truthGammaPool.contains(keyBin)) {
            for (const auto& poolEvent : truthGammaPool[keyBin]) {
              for (const auto& a : trueGammas) {
                for (const auto& b : poolEvent) {
                  fillPairEff<true>(a, b);
                }
              }
            }
          }
        }

        if (doTruthMix) {
          for (size_t i = 0; i < trueGammas.size(); ++i) {
            for (size_t j = i + 1; j < trueGammas.size(); ++j) {
              const auto& g1 = trueGammas[i];
              const auto& g2 = trueGammas[j];
              if (!passAsymmetryCut(g1.pt, g2.pt)) {
                continue;
              }
              const float deta = g1.eta - g2.eta;
              const float dphi = wrapPhi(g1.phi - g2.phi);
              const float px1 = g1.pt * std::cos(g1.phi), py1 = g1.pt * std::sin(g1.phi);
              const float px2 = g2.pt * std::cos(g2.phi), py2 = g2.pt * std::sin(g2.phi);
              const float kt = 0.5f * std::sqrt((px1 + px2) * (px1 + px2) + (py1 + py2) * (py1 + py2));
              const float e1 = g1.pt * std::cosh(g1.eta), e2 = g2.pt * std::cosh(g2.eta);
              const float dot = e1 * e2 - (px1 * px2 + py1 * py2 +
                                           g1.pt * std::sinh(g1.eta) * g2.pt * std::sinh(g2.eta));
              const float qinv_true = std::sqrt(std::max(0.f, 2.f * dot));
              if (qinv_true > mctruth.cfgMCMaxQinv.value) {
                continue;
              }
              fRegistryTruthMC.fill(HIST("MC/TruthCF/hQinvVsKt_same"), kt, qinv_true);
              fRegistryTruthMC.fill(HIST("MC/TruthCF/hDEtaDPhi_same"), deta, dphi);
              if (mctruth.cfgDoTruthLcms.value || mctruth.cfgDoTruth3D.value) {
                const pairutil::PairQ qT = pairutil::computePairQ(ROOT::Math::PtEtaPhiMVector(g1.pt, g1.eta, g1.phi, 0.f),
                                                                  ROOT::Math::PtEtaPhiMVector(g2.pt, g2.eta, g2.phi, 0.f));
                if (mctruth.cfgDoTruthLcms.value) {
                  fRegistryTruthMC.fill(HIST("MC/TruthCF/hSparse_Qout_Qinv_Kt_same"), qT.qout, qinv_true, kt);
                }
                if (mctruth.cfgDoTruth3D.value) {
                  fRegistryTruthMC.fill(HIST("MC/TruthCF/hSparse_Qout_Qside_Qlong_Kt_same"), qT.qout, qT.qside, qT.qlong, kt);
                }
              }
            }
          }

          if (truthGammaPool.contains(keyBin)) {
            for (const auto& poolEvent : truthGammaPool[keyBin]) {
              for (const auto& g1 : trueGammas) {
                for (const auto& g2 : poolEvent) {
                  if (!passAsymmetryCut(g1.pt, g2.pt)) {
                    continue;
                  }
                  const float deta = g1.eta - g2.eta;
                  const float dphi = wrapPhi(g1.phi - g2.phi);
                  const float px1 = g1.pt * std::cos(g1.phi), py1 = g1.pt * std::sin(g1.phi);
                  const float px2 = g2.pt * std::cos(g2.phi), py2 = g2.pt * std::sin(g2.phi);
                  const float kt = 0.5f * std::sqrt((px1 + px2) * (px1 + px2) + (py1 + py2) * (py1 + py2));
                  const float e1 = g1.pt * std::cosh(g1.eta), e2 = g2.pt * std::cosh(g2.eta);
                  const float dot = e1 * e2 - (px1 * px2 + py1 * py2 +
                                               g1.pt * std::sinh(g1.eta) * g2.pt * std::sinh(g2.eta));
                  const float qinv_true = std::sqrt(std::max(0.f, 2.f * dot));
                  if (qinv_true > mctruth.cfgMCMaxQinv.value) {
                    continue;
                  }
                  fRegistryTruthMC.fill(HIST("MC/TruthCF/hQinvVsKt_mix"), kt, qinv_true);
                  fRegistryTruthMC.fill(HIST("MC/TruthCF/hDEtaDPhi_mix"), deta, dphi);
                  if (mctruth.cfgDoTruthLcms.value || mctruth.cfgDoTruth3D.value) {
                    const pairutil::PairQ qT = pairutil::computePairQ(ROOT::Math::PtEtaPhiMVector(g1.pt, g1.eta, g1.phi, 0.f),
                                                                      ROOT::Math::PtEtaPhiMVector(g2.pt, g2.eta, g2.phi, 0.f));
                    if (mctruth.cfgDoTruthLcms.value) {
                      fRegistryTruthMC.fill(HIST("MC/TruthCF/hSparse_Qout_Qinv_Kt_mix"), qT.qout, qinv_true, kt);
                    }
                    if (mctruth.cfgDoTruth3D.value) {
                      fRegistryTruthMC.fill(HIST("MC/TruthCF/hSparse_Qout_Qside_Qlong_Kt_mix"), qT.qout, qT.qside, qT.qlong, kt);
                    }
                  }
                }
              }
            }
          }

        } // end doTruthMix

        if (!trueGammas.empty()) {
          auto& poolBin = truthGammaPool[keyBin];
          poolBin.push_back(trueGammas);
          if (static_cast<int>(poolBin.size()) > mctruth.cfgTruthMixDepth.value) {
            poolBin.pop_front();
          }
        }
      } // end doTruthMix || doPairEff
    } // end collision loop
  } // end runTruthEfficiency

  /*************************************************/
  // Process functions
  /*************************************************/

  void processAnalysis(FilteredMyCollisions const& collisions,
                       MyV0Photons const& v0photons,
                       aod::V0Legs const& v0legs)
  {
    runPairing(collisions, v0photons, v0legs, perCollisionPCM, fV0PhotonCut);
    ndf++;
  }
  PROCESS_SWITCH(Photonhbt, processAnalysis, "pairing for analysis", true);

  void processMC(FilteredMyMCCollisions const& mccollisions,
                 MyV0Photons const& v0photons,
                 MyMCV0Legs const& v0legs,
                 aod::EMMCParticles const& mcParticles,
                 aod::EMMCEvents const& mcEvents)
  {

    runPairingMC(mccollisions, v0photons, v0legs, mcParticles,
                 perCollisionPCM, fV0PhotonCut);
    runTruthEfficiency(mccollisions, v0photons, v0legs, mcParticles, mcEvents, perMCCollisionEMMCParts, perCollisionV0Legs, fV0PhotonCut);

    ndf++;
  }
  PROCESS_SWITCH(Photonhbt, processMC, "MC CF + truth efficiency maps for CF correction", false);

  void processAnalysisAOD(FilteredMyCollisions const& collisions,
                          MyV0Photons const& v0photons,
                          aod::V0Legs const& v0legs,
                          MyTracksIU const& tracks)
  {
    runPairing(collisions, v0photons, v0legs, perCollisionPCM, fV0PhotonCut, &tracks);
    ndf++;
  }
  PROCESS_SWITCH(Photonhbt, processAnalysisAOD, "pairing for analysis on AO2Ds: legs as real tracks in the mixed-event competition", false);

  void processMCAOD(FilteredMyMCCollisions const& mccollisions,
                    MyV0Photons const& v0photons,
                    MyMCV0Legs const& v0legs,
                    aod::EMMCParticles const& mcParticles,
                    aod::EMMCEvents const& mcEvents,
                    MyTracksIU const& tracks)
  {
    runPairingMC(mccollisions, v0photons, v0legs, mcParticles, perCollisionPCM, fV0PhotonCut, &tracks);
    runTruthEfficiency(mccollisions, v0photons, v0legs, mcParticles, mcEvents, perMCCollisionEMMCParts, perCollisionV0Legs, fV0PhotonCut);
    ndf++;
  }
  PROCESS_SWITCH(Photonhbt, processMCAOD, "MC on AO2Ds: as processMC, legs as real tracks in the mixed-event competition", false);

  void processAnalysisXYZ(FilteredMyCollisions const& collisions,
                          MyV0Photons const& v0photons,
                          MyV0LegsXYZ const& v0legs)
  {
    runPairing(collisions, v0photons, v0legs, perCollisionPCM, fV0PhotonCut);
    ndf++;
  }
  PROCESS_SWITCH(Photonhbt, processAnalysisXYZ, "pairing for analysis with V0LegsXYZ: stored and cross candidates through the same analytic builder", false);

  void processMCXYZ(FilteredMyMCCollisions const& mccollisions,
                    MyV0Photons const& v0photons,
                    MyMCV0LegsXYZ const& v0legs,
                    aod::EMMCParticles const& mcParticles,
                    aod::EMMCEvents const& mcEvents)
  {
    runPairingMC(mccollisions, v0photons, v0legs, mcParticles, perCollisionPCM, fV0PhotonCut);
    runTruthEfficiency(mccollisions, v0photons, v0legs, mcParticles, mcEvents, perMCCollisionEMMCParts, perCollisionV0Legs, fV0PhotonCut);
    ndf++;
  }
  PROCESS_SWITCH(Photonhbt, processMCXYZ, "MC with V0LegsXYZ: as processMC, stored and cross candidates through the same analytic builder", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& context)
{
  return WorkflowSpec{adaptAnalysisTask<Photonhbt>(context)};
}
