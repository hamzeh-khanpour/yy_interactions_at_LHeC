#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TLatex.h>
#include <iostream>

void compare_Etall() {



    // Open the ROOT file
    TFile *file = TFile::Open("LHeC_tautau_E_10_10.root");
    if (!file || file->IsZombie()) {
        std::cerr << "Error: Cannot open the ROOT file.\n";
        return;
    }

    // Retrieve the TTrees
    TTree *tree1 = (TTree*)file->Get("LHeC_E");
    TTree *tree2 = (TTree*)file->Get("LHeC_E_750GeV");
    if (!tree1 || !tree2) {
        std::cerr << "Error: One or both TTrees not found in the ROOT file.\n";
        return;
    }


    // Remove statistics globally
    gStyle->SetOptStat(0);


    // Define integrated luminosity and cross-section values
    Float_t integrated_luminosity = 1.0; // fb^-1
    Float_t cross_section_LHeC_E = 4.55272144e+01; // pb
    Float_t cross_section_LHeC_E_750GeV = 3.28684083e+01; // pb

    // Calculate total entries
    Long64_t nentries1 = tree1->GetEntries();
    Long64_t nentries2 = tree2->GetEntries();

    // Calculate event weights
    Float_t event_weight1 = cross_section_LHeC_E * integrated_luminosity / nentries1;
    Float_t event_weight2 = cross_section_LHeC_E_750GeV * integrated_luminosity / nentries2;

    // Create histograms for Etall
    TH1F *hist1 = new TH1F("hist1", "Etall Comparison;#eta^{#tau^{+}};d#sigma/d#eta^{#tau^{+}} [pb/GeV]", 20, -10.0, 10.0);
    TH1F *hist2 = new TH1F("hist2", "Etall Comparison;#eta^{#tau^{+}};d#sigma/d#eta^{#tau^{+}} [pb/GeV]", 20, -10.0, 10.0);

    // Fill histograms with event weights
    tree1->Draw(Form("Etall>>hist1(%d,%f,%f)", 20, -10.0, 10.0), Form("%f", event_weight1));
    tree2->Draw(Form("Etall>>hist2(%d,%f,%f)", 20, -10.0, 10.0), Form("%f", event_weight2));


    // Set histogram styles
    hist1->SetLineColor(kRed);
    hist1->SetLineWidth(2);
    hist1->GetXaxis()->SetTitle("#eta^{#tau^{+}}");
    hist1->GetYaxis()->SetTitle("d#sigma/d#eta^{#tau^{+}} [pb/GeV]");

    hist2->SetLineColor(kBlue);
    hist2->SetLineWidth(2);
    hist2->GetXaxis()->SetTitle("#eta^{#tau^{+}}");
    hist2->GetYaxis()->SetTitle("d#sigma/d#eta^{#tau^{+}} [pb/GeV]");

    // Print integrals
    std::cout << "Integral of Etall (LHeC_E): " << hist1->Integral() << " pb" << std::endl;
    std::cout << "Integral of Etall (LHeC_E_750GeV): " << hist2->Integral() << " pb" << std::endl;

    // Create a canvas for the comparison
    TCanvas *c1 = new TCanvas("c1", "Etall Comparison", 800, 600);

    // Draw histograms
    hist1->Draw("HIST");
    hist2->Draw("HIST SAME");

    // Add a legend
    TLegend *legend = new TLegend(0.6, 0.7, 0.9, 0.9);
    legend->AddEntry(hist1, "elastic (cepgen)", "l");
    legend->AddEntry(hist2, "elastic (cepgen) 0.75 TeV", "l");
    legend->SetTextSize(0.04);
    legend->SetBorderSize(0);
    legend->Draw();

//     // Add additional information to the canvas
     TLatex *label1 = new TLatex();
     label1->SetNDC(); // Normalize coordinates to canvas dimensions
     label1->SetTextSize(0.04);
     label1->DrawLatex(0.15, 0.75, "W > 10 GeV");
//
//     TLatex *label2 = new TLatex();
//     label2->SetNDC();
//     label2->SetTextSize(0.04);
//     label2->SetTextColor(kRed);
//     label2->DrawLatex(0.15, 0.80, "Dataset: LHeC_E");
//
//     TLatex *label3 = new TLatex();
//     label3->SetNDC();
//     label3->SetTextSize(0.04);
//     label3->SetTextColor(kBlue);
//     label3->DrawLatex(0.15, 0.75, "Dataset: LHeC_E_750GeV");

    // Save the canvas
    c1->SaveAs("Etall_Comparison_Weighted.pdf");
    c1->SaveAs("Etall_Comparison_Weighted.png");
    c1->SaveAs("Etall_Comparison_Weighted.root");

    // Clean up
    delete hist1;
    delete hist2;

     delete label1;
//     delete label2;
//     delete label3;

    delete legend;
    delete c1;
    file->Close();
}
