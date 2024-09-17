/*
Create a first Training Dataset:
Use as truth labels the hits in tracks belonging to
positron events
Use as Mom, Phi and Theta of the track the informations at the target.
(Optionally) Smear time information of hits.

SPX hits ID in the dataset is augmented by 1920.
Layer is 10.
x, y, z from pixelID position.
phi and theta wire can not be provided->Let's start not using this information
at the beginning.
*/

#include <TROOT.h>
#include <TEfficiency.h>
#include <TMath.h>
#include <TStyle.h>
#include <TChain.h>
#include <TGraph.h>
#include <TCanvas.h>
#include <TFile.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <iostream>
#include <fstream>
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
#   include "cyldch/MEGCYLDCHGeometry.h"
#else
   class ROMETreeInfo;
#endif
using namespace MEG;
using namespace std;

vector<int> wireIDSorting(vector<double> values, vector<int> ids) {

  // initialize original index locations
  vector<size_t> idx(values.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
	      [&values](size_t i1, size_t i2) {return values[i1] < values[i2];});

  vector<int> copy_vector; 
  for (auto i : idx) {
    copy_vector.push_back(ids.at(i));
  }
  return copy_vector;

}

vector<double> zSorting(vector<double> values, vector<int> ids) {

  // initialize original index locations
  vector<size_t> idx(values.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
	      [&values](size_t i1, size_t i2) {return values[i1] < values[i2];});

  vector<double> copy_vector;
  for (auto i : idx) {
    copy_vector.push_back(values.at(i));
  }
  return copy_vector;

}

vector<double> phiSorting(vector<double> values, vector<double> zValues) {

  // initialize original index locations
  vector<size_t> idx(values.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
	      [&zValues](size_t i1, size_t i2) {return zValues[i1] < zValues[i2];});

  vector<double> copy_vector;
  for (auto i : idx) {
    copy_vector.push_back(values.at(i));
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


//______________________________________________________________________________
vector<string> split(const string& input, char delimiter)
{
    istringstream stream(input);

    string field;
    vector<string> result;
    while (getline(stream, field, delimiter)) {
        result.push_back(field);
    }
    return result;
}

//______________________________________________________________________________
void WriteTrainingData()
{
  
  // Train, Test and Val output files
  TString outputrecdir = "./"; //"/meg/data1/shared/subprojects/cdch/ext-venturini_a/GNN/";
  TString outputfilename_train = outputrecdir + "1e6TrainSet_CDCH.txt";
  ofstream outputfile_train;
  outputfile_train.open(outputfilename_train.Data());
  TString outputfilename_test = outputrecdir + "1e6TestSet_CDCH.txt";
  ofstream outputfile_test;
  outputfile_test.open(outputfilename_test.Data());
  TString outputfilename_val = outputrecdir + "1e6ValSet_CDCH.txt";
  ofstream outputfile_val;
  outputfile_val.open(outputfilename_val.Data());

  int nMaxEvents = 100000;
  double f_train = 0.7;
  double f_test = 0.15;

  TString inputrecdir = "/meg/data1/offline/processes/20240209/425xxx/"; // "/meg/data1/shared/subprojects/cdch/ext-venturini_a/2021_3e7/";
  TString runList = "";
  Int_t sRun = 0;
  Int_t nfile = 999;
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
  TBranch *bSPXHit;
  TClonesArray *pSPXHitArray = new TClonesArray("MEGSPXHit");
  TBranch *bDCHTrack;
  TClonesArray *pDCHTrackArray = new TClonesArray("MEGDCHTrack");
  TBranch *bDCHHit;
  TClonesArray *pDCHHitArray = new TClonesArray("MEGDCHHit");
  TBranch *bDCHHitRecResult;
  TClonesArray *pDCHHitRecResultArray = new TClonesArray("MEGCYLDCHHitRecResult");
  TBranch *bSPXTrack;
  TClonesArray *pSPXTrackArray = new TClonesArray("MEGSPXTrack");
  TBranch *bDCHSPXMatchedTrack;
  TClonesArray *pDCHSPXMatchedTrackArray = new TClonesArray("MEGDCHSPXMatchedTrack");
  
  TBranch *bInfoS;
  ROMETreeInfo *pInfoS;
  MEGTargetRunHeader *pTargetRunHeader = (MEGTargetRunHeader*)gROOT->FindObject("TargetRunHeader");
  // Setting up geometry                                                                  
  auto pMEGParameters = (MEGMEGParameters*)(file0->Get("MEGParameters"));               
  auto pWireRunHeaders = static_cast<TClonesArray*>(file0->Get("CYLDCHWireRunHeader")); 
  auto fCYLDCHGeometry = MEGCYLDCHGeometry::GetInstance(pWireRunHeaders);

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
  rec->SetBranchStatus("dchhitrecresults", 1);
  rec->SetBranchStatus("dchhitrecresults.*", 1);
  rec->SetBranchStatus("spxclusters", 1); 
  rec->SetBranchStatus("spxclusters.*", 1); 
  rec->SetBranchStatus("spxhits", 1); 
  rec->SetBranchStatus("spxhits.*", 1); 
  rec->SetBranchStatus("dchspxmatchedtrack", 1); 
  rec->SetBranchStatus("dchspxmatchedtrack.*", 1); 
  rec->SetBranchStatus("spxtracks", 1); 
  rec->SetBranchStatus("spxtracks.*", 1); 
   
  rec->SetBranchAddress("Info.", &pInfoS, &bInfoS);
  rec->SetBranchAddress("eventheader.", &pEventHeader, &bEventHeader);
  rec->SetBranchAddress("positron", &pPositronArray, &bPositron);
  rec->SetBranchAddress("spxclusters", &pSPXClusterArray, &bSPXCluster);
  rec->SetBranchAddress("spxhits", &pSPXHitArray, &bSPXHit);
  rec->SetBranchAddress("dchspxmatchedtrack", &pDCHSPXMatchedTrackArray, &bDCHSPXMatchedTrack);
  rec->SetBranchAddress("dchtracks", &pDCHTrackArray, &bDCHTrack);
  rec->SetBranchAddress("dchhits", &pDCHHitArray, &bDCHHit);
  rec->SetBranchAddress("dchhitrecresults", &pDCHHitRecResultArray, &bDCHHitRecResult);
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
  
  selector.fPrePositronSelectTimePositronGamma[0] = denominatorCutTC.timeWindow[0] - refTime;
  selector.fPrePositronSelectTimePositronGamma[1] = denominatorCutTC.timeWindow[1] - refTime;
  selector.fEPositron[0] = 42 * MeV;
  selector.fEPositron[1] = 56 * MeV;
  
  selector.fSPXNHitsForEfficiencyEstimation = denominatorCutTC.nClusterHitThreshold;

  selector.fTargetZOffset = targetOffset[2];
  selector.fTargetYOffset = targetOffset[1];
  selector.fTargetZ = targetSemiAxis[0];
  selector.fTargetY = targetSemiAxis[1];
  
  for (Int_t iEvent = 0; iEvent < max(nMaxEvents, nEvent); iEvent++) {
    
    if (iEvent % 5000 == 1) {
      cout<<iEvent<<" events finished..."<<endl;
    } 
    rec->GetEntry(iEvent);
    
    // Trigger
    auto mask = pEventHeader->Getmask();
    if (triggerMask.find(mask) == triggerMask.end()) continue;
    
    if (!pPositronArray || !pSPXTrackArray || !pDCHTrackArray || !pDCHHitArray) continue;
    
    // Select positrons
    vector<Bool_t> selected;
    Bool_t oriSelectAPositron = selector.fSelectAPositron;
    selector.fSelectAPositron = kFALSE;
    selector.PositronSelection(selected, pPositronArray, false); 
    
    vector<vector<int>> hits_id;
    vector<int> idx_positron;
    vector<double> mom_target;
    vector<double> phi_target;
    vector<int> theta_target;

    // Loop over selected positrons
    int npos = 0;
    for (Int_t ipos=0; ipos < pPositronArray->GetEntriesFast(); ipos++) {
      
      /*
      if (!selected[ipos]) {
	continue;
      }
      */
      
      MEGGLBPositron* pPositron = (MEGGLBPositron*)pPositronArray->At(ipos);
      MEGDCHTrack* pDCHTrack = (MEGDCHTrack*)pDCHTrackArray->At(pPositron->GetDCHTrackIndex());
      MEGSPXTrack* pSPXTrack = (MEGSPXTrack*)pSPXTrackArray->At(pPositron->GetSPXTrackIndex());
      MEGDCHSPXMatchedTrack* pDCHSPXMatchedTrack = (MEGDCHSPXMatchedTrack*)pDCHSPXMatchedTrackArray->At(pPositron->GetDCHTrackIndex());
      
      if (pPositron->GetDCHTrackIndex() != pDCHSPXMatchedTrack->GetMatchedTrackIdAt(0)) {
	cout << "Not matching tracks" << endl;
	continue;
      }
      
      npos++;

      // Fill track info
      idx_positron.push_back(ipos);
      MEGStateVector* stateTarget = (MEGStateVector*)pDCHTrack->GetStateVectorTarget();
      mom_target.push_back(stateTarget->GetP());
      phi_target.push_back(stateTarget->GetPhi());
      theta_target.push_back(stateTarget->GetTheta());

      // Loop over hits in the DCH Track to have good hit index
      Int_t nhits = pDCHTrack->Getnhits();
      vector<int> hit_idx;
      vector<double> z_vec;

      for (Int_t ihit=0; ihit < nhits; ihit++) {
	
	// Skip hits not used in the fit
	if (!pDCHTrack->GetHasPredictionAt(ihit)) {
	  continue;
	}
	
	MEGDCHHit* aHit = (MEGDCHHit*)pDCHHitArray->At(pDCHTrack->GethitindexAt(ihit));
	if (!aHit->Getgood()) {
	  continue;
	}
	MEGStateVector* state = (MEGStateVector*)pDCHTrack->GetPredictedStateVectorAt(ihit);
	hit_idx.push_back(pDCHTrack->GethitindexAt(ihit));
	z_vec.push_back(state->GetZ());
      }
      /*
      // Do the same on spx hits
      Int_t nhits_spx = pSPXTrack->Getnhits();
      
      for (Int_t ihit=0; ihit < nhits_spx; ihit++) {
	
	// Skip hits not used in the fit
	if (pSPXTrack->GetSkippedAt(ihit)) {
	  continue;
	}
	MEGSPXHit* aHit = (MEGSPXHit*)pSPXHitArray->At(pSPXTrack->GethitindexAt(ihit));
	hit_idx.push_back(pSPXTrack->GethitindexAt(ihit) + 1920);
      }
      */

      // Fill vector with index from this track after sorting
      vector<int> sorted_idx = wireIDSorting(z_vec, hit_idx);
      hits_id.push_back(sorted_idx);
      
    } // Positron Loop
   
    if (npos == 0) continue;

    // Loop over all DCH Hits and then over all SPX Hits to write the training file
    for (int i=0; i<pDCHHitArray->GetEntriesFast(); i++) {
      MEGDCHHit* aHit = (MEGDCHHit*)pDCHHitArray->At(i);
      if (!aHit) {
	cout << "No hit at idx i = " << i << endl;
	continue;
      }
      double z = aHit->Getztimediff();
      // Cut on badly reconstructed hits
      if (abs(z) > 200 ) continue;

      // Get all info of the hit
      // Signal informations
      double charge0 = -1;
      double ampl0 = -1;
      double charge1 = -1;
      double ampl1 = -1;
      
      if (aHit->GetHitRecResultIndexAt(0) >= 0) {
	MEGCYLDCHHitRecResult* aHitRes0 = (MEGCYLDCHHitRecResult*)pDCHHitRecResultArray->At(aHit->GetHitRecResultIndex()[0]);
	if (aHitRes0) {
	  charge0 = aHitRes0->Getcharge();
	  ampl0 = aHitRes0->GetamplsignFullRange();
	}
      }
      if (aHit->GetHitRecResultIndexAt(1) >= 0) {
	MEGCYLDCHHitRecResult* aHitRes1 = (MEGCYLDCHHitRecResult*)pDCHHitRecResultArray->At(aHit->GetHitRecResultIndex()[1]);
	if (aHitRes1) {
	  charge1 = aHitRes1->Getcharge();
	  ampl1 = aHitRes1->GetamplsignFullRange();
	}
      }
      
      // Geometry
      int wire = aHit->Getwire();
      double time = aHit->Gettime();
      int layer = aHit->Getplane();
      TVector3 wireVec = {0., 0., 0.};
      wireVec = fCYLDCHGeometry->LocalToGlobal(wire, wireVec);
      double x0 = wireVec.X();
      double y0 = wireVec.Y();
      double z0 = wireVec.Z();
      double sigmaz = (abs(z) > 0.5) ? 10 : 200; // if z is not reconstructed, give huge uncertainty
      TVector3 p0;
      TVector3 uaxis;
      TVector3 vaxis;
      TVector3 waxis;
      fCYLDCHGeometry->GetLocalFrame(wire, 0., p0, uaxis, vaxis, waxis);
      double phi = waxis.Phi();
      double theta = waxis.Theta();
      // Belongs to track
      int is_good = 0; // not good
      int track_id = -1; // not good
      double mom = 1e30; // not good
      double phitrk = 1e30; // not good
      double thetatrk = 1e30; // not good
      int next_hit_idx = -1; // not good
      // Check if it is a good hit
      int track_idx = 0;
      for (auto ids : hits_id) {
	auto found_idx = find(ids.begin(), ids.end(), i); 
	if (found_idx != ids.end()) {
	  // Get next hit
	  int found_hit_idx = found_idx - ids.begin();
	  next_hit_idx = (found_hit_idx == (int)(ids.size() - 1)) ? -1 : ids.at(found_hit_idx + 1);
	  is_good = 1;
	  track_id = track_idx;
	  mom = mom_target.at(track_idx);
	  phitrk = phi_target.at(track_idx);
	  thetatrk = theta_target.at(track_idx);
	  break;
	}
	track_idx++;
      }

      // write
      if (iEvent < f_train * max(nMaxEvents, nEvent)) {
	outputfile_train << i << " " << wire << " " << time << " " << layer << " " << charge0 << " " << charge1 << " " << ampl0 << " " << ampl1 << " " << x0 << " " << y0 << " " << z0 << " " << z << " " << sigmaz << " " << phi << " " << theta << " " << is_good << " " << next_hit_idx << " "  << track_id << " " << mom << " " << phitrk << " " << thetatrk << endl;
      }
      else if (iEvent < (f_train + f_test) * max(nMaxEvents, nEvent)) {
	outputfile_test << i << " " << wire << " " << time << " " << layer << " " << charge0 << " " << charge1 << " " << ampl0 << " " << ampl1 << " " << x0 << " " << y0 << " " << z0 << " " << z << " " << sigmaz << " " << phi << " " << theta << " " << is_good << " " << next_hit_idx << " "  << track_id << " " << mom << " " << phitrk << " " << thetatrk << endl;
      }
      else {
	outputfile_val << i << " " << wire << " " << time << " " << layer << " " << charge0 << " " << charge1 << " " << ampl0 << " " << ampl1 << " " << x0 << " " << y0 << " " << z0 << " " << z << " " << sigmaz << " " << phi << " " << theta << " " << is_good << " " << next_hit_idx << " "  << track_id << " " << mom << " " << phitrk << " " << thetatrk << endl;
      }

    }
  /*
    for (int i=0; i<pSPXHitArray->GetEntriesFast(); i++) {
      MEGSPXHit* aHit = (MEGSPXHit*)pSPXHitArray->At(i);
      if (!aHit) {
	cout << "No hit at idx i = " << i << endl;
	continue;
      }
      // Get all info of the hit
      int pixel = aHit->Getpixelid() + 1920;
      double time = aHit->Gettime();
      int layer = 10;
      double x = aHit->GetxyzAt(0);
      double y = aHit->GetxyzAt(1);
      double z = aHit->GetxyzAt(2);
      int is_good = 0; // not good
      int track_id = -1; // not good
      double mom = 1e30; // not good
      double phi = 1e30; // not good
      double theta = 1e30; // not good

      // Check if it is a good hit
      int track_idx = 0;
      for (auto ids : hits_id) {
	if (find(ids.begin(), ids.end(), i) != ids.end()) {
	  is_good = 1;
	  track_id = track_idx;
	  mom = mom_target.at(track_idx);
	  phi = phi_target.at(track_idx);
	  theta = theta_target.at(track_idx);
	  break;
	}
	track_idx++;
      }

      // write
      if (iEvent < f_train * min(nMaxEvents, nEvent)) {
	outputfile_train << i + pDCHHitArray->GetEntriesFast() << " " << pixel << " " << time << " " << layer << " " << x << " " << y << " " << z << " " << is_good << " " << track_id << " " << mom << " " << phi << " " << theta << endl;
      }
      else if (iEvent < (f_train + f_test) * min(nMaxEvents, nEvent)) {
	outputfile_test << i + pDCHHitArray->GetEntriesFast() << " " << pixel << " " << time << " " << layer << " " << x << " " << y << " " << z << " " << is_good << " " << track_id << " " << mom << " " << phi << " " << theta << endl;
      }
      else {
	outputfile_val << i + pDCHHitArray->GetEntriesFast() << " " << pixel << " " << time << " " << layer << " " << x << " " << y << " " << z << " " << is_good << " " << track_id << " " << mom << " " << phi << " " << theta << endl;
      }

    }
  */

  } // Event Loop
  
  // Close files
  cout << outputfilename_train << " written!" << endl;
  outputfile_train.close();
  cout << outputfilename_test << " written!" << endl;
  outputfile_test.close();
  cout << outputfilename_val << " written!" << endl;
  outputfile_val.close();

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
         if (file.find("rec") != string::npos
             && file.find(".root") != string::npos) {
            smatch results;
            if (regex_match(file, results, regex("rec(\\d+).root"))) {
               files[atoi(results[1].str().data())] = dir + file.c_str();
            }
         }
      }
      if (closedir(pDir) != 0) {
         perror("closedir");
         return files;
      }

   } else {
      // Read a run list

      ifstream runListFile(runlist);
      string str;
      vector<char> delimiters {'\t', ',', ' '};
      while (getline(runListFile, str)) {
         for (auto && deli: delimiters) {
            if (str.find(deli) != string::npos) {
               auto a = split(str, deli);
               str = a[0];
               break;
            }
         }
         
         Int_t run = -1;
         try {
            run = stoi(str);
         } catch (const invalid_argument& e) {
            //cout << "[" << i << "]: " << "invalid argument" << endl;
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
         ifstream ifs(recfile.Data());
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
