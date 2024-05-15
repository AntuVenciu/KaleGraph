#include <TROOT.h>
#include <TEfficiency.h>
#include <TMath.h>
#include <TStyle.h>
#include <TChain.h>
#include <TFile.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <map>
#include <set>
#include <regex>
#include <dirent.h>
#include <sys/stat.h>
#include "../common/include/units/MEGSystemOfUnits.h"
#include "units/MEGPhysicalConstants.h"
#if !defined(__CLING__) || defined(__ROOTCLING__)
#   include <ROMEString.h>
#   include <ROMETreeInfo.h>
#   include "include/generated/MEGAllFolders.h"
#   include "glb/MEGPhysicsSelection.h"
#else
   class ROMETreeInfo;
#endif
using namespace MEG;
using namespace std;


vector<int> wireIDSorting(vector<double> values, vector<int> ids) {

  // initialize original index locations
  vector<size_t> idx(values.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  std::stable_sort(idx.begin(), idx.end(),
		   [&values](size_t i1, size_t i2) {return values[i1] < values[i2];});

  std::vector<int> copy_vector;
  for (auto i : idx) {
    copy_vector.push_back(ids.at(i));
  }
  return copy_vector;

}

map<Int_t, TString> PrepareRunList(TString dir, TString list, Int_t startRun=0, Int_t maxNRuns=-1);

struct SPXDenominatorCut {
   Double_t timeWindow[2];  // efficiency vs time
   Double_t phiWindowTC[2]; // efficiency vs phi
   Double_t zWindowTC[2];   // efficiency vs z
   Short_t nClusterHitThreshold; // Apply clusters.nhits >= nClusterHitThreshold cut for denominator
};

tuple<Int_t, Int_t, Int_t>  TrackingEfficiencyStudy(
      MEGPhysicsSelection& selector,
      SPXDenominatorCut& spxdenominatorCut,
      const TClonesArray* spxclusters,
      const TClonesArray* spxindependenttracks,      
      const TClonesArray* positrons,
      const TClonesArray* spxtracks,
      Double_t refTime
);

//______________________________________________________________________________
std::vector<std::string> split(const std::string& input, char delimiter)
{
    std::istringstream stream(input);

    std::string field;
    std::vector<std::string> result;
    while (std::getline(stream, field, delimiter)) {
        result.push_back(field);
    }
    return result;
}



//______________________________________________________________________________
void DataDrivenEdgeConnectionStudy()
{

  TString inputrecdir = "/meg/data1/offline/processes/20240209/424xxx/"; // "/meg/data1/shared/subprojects/cdch/ext-venturini_a/2021_3e7/";
   TString runList = "";
   Int_t sRun = 0;
   Int_t nfile = 205;
   set<Int_t> triggerMask {20};

   // Apply selection to denominator of the efficiency evaluation
   // TC clusters that passes all the time, z and phi selection are accounted for the efficiency evaluation
   // If you do not need to apply selections to some of the parameters, just set loose enough values.
   SPXDenominatorCut denominatorCutTC;
   // for Efficiency vs time study
   denominatorCutTC.timeWindow[0] = -642 * nanosecond;
   denominatorCutTC.timeWindow[1] = -600 * nanosecond;
   // for Efficiency vs zTC study
   denominatorCutTC.zWindowTC[0] = -200 * centimeter;
   denominatorCutTC.zWindowTC[1] = 200 * centimeter;
   // for Efficiency vs phiTC study
   denominatorCutTC.phiWindowTC[0] = -180 * degree;
   denominatorCutTC.phiWindowTC[1] = 20 * degree;
   denominatorCutTC.nClusterHitThreshold = 4; // Require >= 4 hits

   

   // map of input rec files
   map<Int_t, TString> files = PrepareRunList(inputrecdir, runList, sRun, nfile);
   Int_t startRun = files.begin()->first;
   Int_t lastRun = files.rbegin()->first;

   
   // Open the files
   TFile *file0 = 0;
   TChain *rec = new TChain("rec");
   for (auto &[run, file] : files) {
      if (gSystem->AccessPathName(file)) continue;
      rec->Add(file);
      if (!file0) {
         file0 = TFile::Open(file);
      }
   }



   TBranch *bEventHeader;
   MEGEventHeader *pEventHeader;
   TBranch *bPositron;
   TClonesArray *pPositronArray = new TClonesArray("MEGGLBPositron");
   TBranch *bSPXCluster;
   TClonesArray *pSPXClusterArray = new TClonesArray("MEGSPXCluster");
   TBranch *bDCHTrack;
   TClonesArray *pDCHTrackArray = new TClonesArray("MEGDCHTrack");
   TBranch *bDCHHit;
   TClonesArray *pDCHHitArray = new TClonesArray("MEGDCHHit");
   TBranch *bSPXTrack;
   TClonesArray *pSPXTrackArray = new TClonesArray("MEGSPXTrack");
   
   TBranch *bInfoS;
   ROMETreeInfo *pInfoS;
   MEGTargetRunHeader *pTargetRunHeader = (MEGTargetRunHeader*)gROOT->FindObject("TargetRunHeader");
   pInfoS = new ROMETreeInfo;
   pEventHeader = new MEGEventHeader;
   rec->SetBranchStatus("*", 0); 
   rec->SetBranchStatus("Info.", 1); 
   rec->SetBranchStatus("Info.*", 1); 
   rec->SetBranchStatus("eventheader.", 1); 
   rec->SetBranchStatus("eventheader.mask", 1);
   rec->SetBranchStatus("positron", 1);
   rec->SetBranchStatus("positron.*", 1);
   rec->SetBranchStatus("dchtracks", 1);
   rec->SetBranchStatus("dchtracks.*", 1);
   rec->SetBranchStatus("dchhits", 1);
   rec->SetBranchStatus("dchhits.*", 1);
   rec->SetBranchStatus("spxclusters", 1); 
   rec->SetBranchStatus("spxclusters.*", 1); 
   //rec->SetBranchStatus("spxindependenttracks", 1); 
   //rec->SetBranchStatus("spxindependenttracks.*", 1); 
   rec->SetBranchStatus("spxtracks", 1); 
   rec->SetBranchStatus("spxtracks.*", 1); 
   
   rec->SetBranchAddress("Info.", &pInfoS, &bInfoS);
   rec->SetBranchAddress("eventheader.", &pEventHeader, &bEventHeader);
   rec->SetBranchAddress("positron", &pPositronArray, &bPositron);
   rec->SetBranchAddress("spxclusters", &pSPXClusterArray, &bSPXCluster);
   //rec->SetBranchAddress("spxindependenttracks", &pSPXIndependentTrackArray, &bSPXIndependentTrack);
   rec->SetBranchAddress("dchtracks", &pDCHTrackArray, &bDCHTrack);
   rec->SetBranchAddress("dchhits", &pDCHHitArray, &bDCHHit);
   rec->SetBranchAddress("spxtracks", &pSPXTrackArray, &bSPXTrack);


   Double_t targetSlantAngle = pTargetRunHeader->GetEulerAnglesAt(1);
   TVector3 targetOffset(pTargetRunHeader->GetTargetPosition());
   Double_t targetSemiAxis[2] = {pTargetRunHeader->GetLongAxis() / 2. 
                                 * TMath::Cos(targetSlantAngle*TMath::DegToRad()),
                                 pTargetRunHeader->GetShortAxis() / 2. };

   
   Int_t nEvent = rec->GetEntries(); 
   cout<<"Number of Event "<<nEvent<<endl;

   Double_t refTime = (denominatorCutTC.timeWindow[0] + denominatorCutTC.timeWindow[1]) / 2;
   // Setup MEGPhysicsSelection
   MEGPhysicsSelection selector(kFALSE, 0, kTRUE);
   selector.SetThresholds(EBeamPeriodID::kBeamPeriod2022, kTRUE);

   // update threshoulds and parameters to custom values
   /*
   selector.fPrePositronSelectTimePositronGamma[0] = denominatorCutTC.timeWindow[0] - refTime;
   selector.fPrePositronSelectTimePositronGamma[1] = denominatorCutTC.timeWindow[1] - refTime;
   selector.fEPositron[0] = 42 * MeV; //42
   selector.fEPositron[1] = 56 * MeV;

   selector.fSPXNHitsForEfficiencyEstimation = denominatorCutTC.nClusterHitThreshold;

   selector.fTargetZOffset = targetOffset[2];
   selector.fTargetYOffset = targetOffset[1];
   selector.fTargetZ = targetSemiAxis[0];
   selector.fTargetY = targetSemiAxis[1];
   */

   int EdgeMatrix[1920][1920];

   for (Int_t iEvent = 0; iEvent < nEvent; iEvent++) {
      
      if (iEvent % 5000 == 1) {
         cout<<iEvent<<" events finished..."<<endl;
      } 
      rec->GetEntry(iEvent);

      // Trigger
      auto mask = pEventHeader->Getmask();
      if (triggerMask.find(mask) == triggerMask.end()) continue;
      
      if (!pPositronArray || !pSPXTrackArray) continue;

      // Select positrons
      vector<Bool_t> selected;
      Bool_t oriSelectAPositron = selector.fSelectAPositron;
      selector.fSelectAPositron = kFALSE;
      selector.PositronSelection(selected, pPositronArray, false); 

      // Loop over selected positrons
      for (Int_t ipos=0; ipos < pPositronArray->GetEntriesFast(); ipos++) {

	if (!selected[ipos]) {
	  continue;
	}
	
	MEGGLBPositron* pPositron = (MEGGLBPositron*)pPositronArray->At(ipos);
	MEGDCHTrack* pDCHTrack = (MEGDCHTrack*)pDCHTrackArray->At(pPositron->GetDCHTrackIndex());

	// Loop over hits in the DCH Track and sort them.
	// Then, we fill the EdgeMatrix
	
	Int_t nhits = pDCHTrack->Getnhits();
	std::vector<double> zhit_vec;
	std::vector<Int_t> wireID_vec;

	for (Int_t ihit=0; ihit < nhits; ihit++) {

	  // Skip hits not used in the fit
	  if (!pDCHTrack->GetHasPredictionAt(ihit)) {
	    continue;
	  }

	  MEGDCHHit* aHit = (MEGDCHHit*)pDCHHitArray->At(pDCHTrack->GethitindexAt(ihit));
	  Int_t wireID = aHit->Getwire();
	  wireID_vec.push_back(wireID);
	  MEGStateVector* state = (MEGStateVector*) pDCHTrack->GetPredictedStateVectorAt(ihit);
	  double zhit = state->GetZ();
	  zhit_vec.push_back(zhit);
	}

	// Sorting
	vector<int> wireID_sorted = wireIDSorting(zhit_vec, wireID_vec);

	// Filling EdgeMatrix
	for (Int_t id=0; id < wireID_sorted.size() - 1; id++) {
	  EdgeMatrix[wireID_sorted.at(id)][wireID_sorted.at(id + 1)]++;
	}

      }

   } // Event Loop


}

//______________________________________________________________________________
map<Int_t, TString> PrepareRunList(TString dir, TString runlist, Int_t startRun, Int_t maxNRuns)
{
   // Return map of run list (runnumber, file path)
   // If list is empty, get all rec files in the dir



   // map of input rec files
   map<Int_t, TString> files;

   if (!runlist.Length()) {
      // Get all the files in a directory
      auto *pDir = opendir(dir.Data());
      if (pDir == nullptr) {
         perror("opendir");
         return files;
      }
      dirent *pDirEnt = nullptr;
      while ((pDirEnt = readdir(pDir)) != nullptr) {
         string file = pDirEnt->d_name;
         if (file.find("rec") != std::string::npos
             && file.find(".root") != std::string::npos) {
            std::smatch results;
            if (std::regex_match(file, results, std::regex("rec(\\d+).root"))) {
               files[std::atoi(results[1].str().data())] = dir + file.c_str();
            }
         }
      }
      if (closedir(pDir) != 0) {
         perror("closedir");
         return files;
      }

   } else {
      // Read a run list

      std::ifstream runListFile(runlist);
      std::string str;
      vector<char> delimiters {'\t', ',', ' '};
      while (std::getline(runListFile, str)) {
         for (auto && deli: delimiters) {
            if (str.find(deli) != std::string::npos) {
               auto a = split(str, deli);
               str = a[0];
               break;
            }
         }
         
         Int_t run = -1;
         try {
            run = std::stoi(str);
         } catch (const std::invalid_argument& e) {
            //std::cout << "[" << i << "]: " << "invalid argument" << std::endl;
            continue;
         }
         
         struct stat sta;
         TString recfile;
         // check subdir exists
         if (!stat(Form("%s/%03dxxx", dir.Data(), run /1000), &sta)) {
            recfile = dir + Form("%03dxxx/rec%05d.root", run / 1000, run);
         } else {
            recfile = dir + Form("rec%05d.root", run);
         }
         std::ifstream ifs(recfile.Data());
         if (!ifs.is_open()) continue;
         ifs.close();
         
         files[run] = recfile.Data();
      }
   }



   Int_t nAdded(0);
   for (auto it = files.begin(); it != files.end();) {
      if (it->first < startRun || (maxNRuns>0 && ++nAdded >= maxNRuns)) {
         it = files.erase(it);
      } else {
         ++it;
      }
   }
   
   return files;
}


tuple<Int_t, Int_t, Int_t>  TrackingEfficiencyStudy(
      MEGPhysicsSelection& selector,
      SPXDenominatorCut& spxdenominatorCut,
      const TClonesArray* spxclusters,
      const TClonesArray* spxindependenttracks,
      const TClonesArray* positrons,
      const TClonesArray* spxtracks,
      Double_t refTime
                                                                  )
{
   // spxdenominatorCut is to apply cuts on TC hit time and/or position
   // To be used for tracking efficiency estimation.
   // Return the number of tracks for denominator and numerator.

   Int_t denominator(0), numerator(0);
   Int_t nClusterWholeRegion(0); // number of TC clusters without position cut (same as denominator if you choose not to apply z/phi cut)

   if (!positrons || !spxtracks) {
      Report(R_ERROR, "Bad input");
      return {denominator, numerator, nClusterWholeRegion};
   }

   Bool_t useSPXIndependentTrack = kFALSE;
   if ((useSPXIndependentTrack && !spxindependenttracks)
       || (!useSPXIndependentTrack && !spxclusters)){
      Report(R_ERROR, "Bad input");
      return {denominator, numerator, nClusterWholeRegion};
   }      

   
   // Count denominator
   set<Int_t> denoSPXClusterIndeces;
   if (!useSPXIndependentTrack) { // SPXCluster base
      Int_t nSPXClusters = spxclusters->GetEntriesFast();
      for (Int_t iSPXCluster = 0; iSPXCluster < nSPXClusters; iSPXCluster++) {
         auto pSPXCluster = (MEGSPXCluster*)spxclusters->At(iSPXCluster);
         // Use only the first cluster in a cluster-group
         if (pSPXCluster->GetindexInGroup() != 0) {
            continue;
         }
         // Nhits selection
         if (pSPXCluster->Getnhits() < selector.fSPXNHitsForEfficiencyEstimation) {
            continue;
         }
         // Timing selection
         Double_t timediff = pSPXCluster->Gettime() - refTime;
         if (timediff < selector.fPrePositronSelectTimePositronGamma[0]
             || timediff > selector.fPrePositronSelectTimePositronGamma[1]) {
            continue;
         }
         ++nClusterWholeRegion;
         // z position selection
         Double_t zTCCluster = pSPXCluster->Getz();
         if (zTCCluster < spxdenominatorCut.zWindowTC[0]
             || zTCCluster > spxdenominatorCut.zWindowTC[1]) {
            continue;
         }
         Double_t phiTCCluster = pSPXCluster->Getphi();
         if (phiTCCluster < spxdenominatorCut.phiWindowTC[0]
             || phiTCCluster > spxdenominatorCut.phiWindowTC[1]) {
            continue;
         }
         
         ++denominator;
         denoSPXClusterIndeces.insert(iSPXCluster);
      }
   } else { // SPXIndependentTrack base
      // To be implemented
   }

   // Count numerator
   set<Int_t> matchedSPXClusterIndeces; // set of SPXCluster already matched to DCHTrack
   vector<Bool_t> selected;
   Bool_t oriSelectAPositron = selector.fSelectAPositron;
   selector.fSelectAPositron = kFALSE;
   selector.PositronSelection(selected, positrons, false); 

   Int_t nGLBPositrons = positrons->GetEntriesFast();
   //cout << "positrons " << nGLBPositrons << endl;
   for (Int_t iGLBPositron = 0; iGLBPositron < nGLBPositrons; iGLBPositron++) {
      if (!selected[iGLBPositron]) {
         continue;
      }
      auto pGLBPositron = (MEGGLBPositron*)positrons->At(iGLBPositron);

      // Check the correspondence with the SPXClusters in denominator
      Int_t spxTrackIndex = pGLBPositron->GetSPXTrackIndex();
      //cout << "spx track index = " << spxTrackIndex << endl;
      if (spxTrackIndex >= spxtracks->GetEntriesFast()) continue;
      auto pSPXTrack = (MEGSPXTrack*)spxtracks->At(spxTrackIndex);
      Int_t clusterIndex = pSPXTrack->GetclusterindexAt(0);
      if (denoSPXClusterIndeces.find(clusterIndex) != denoSPXClusterIndeces.end() &&
          matchedSPXClusterIndeces.find(clusterIndex) == matchedSPXClusterIndeces.end()) { 
         matchedSPXClusterIndeces.insert(clusterIndex);
         ++numerator;
      }
   }
   
   // revert setting
   selector.fSelectAPositron = oriSelectAPositron;
   return {denominator, numerator, nClusterWholeRegion};
}
